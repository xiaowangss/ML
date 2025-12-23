# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor  # 替换为更快的ExtraTrees
from sklearn.preprocessing import RobustScaler

# 记录程序开始时间
start_time = time.time()

# ---------------------- 核心配置 ----------------------
TIME_COLUMN = 'time'
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
CL = columns + noise_columns


# ---------------------- 数据加载 ----------------------
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=[TIME_COLUMN])
    except:
        df = pd.read_csv(file_path)
    return df


# 路径请保持原样
train_path = r'D:\ML期末代码比赛\加噪数据集\加噪数据集\modified_数据集Time_Series661_detail.dat'
test_path = r'D:\ML期末代码比赛\加噪数据集\加噪数据集\modified_数据集Time_Series662_detail.dat'

train_dataSet = load_data(train_path)
test_dataSet = load_data(test_path)


# ---------------------- 强化版数据预处理 ----------------------
def preprocess_data_enhanced(df, is_train=True, scaler=None):
    data = df.copy()

    # 1. 缺失值处理：线性插值（比中位数更适合时间序列）
    # 仅针对数值列进行插值
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].interpolate(method='linear', limit_direction='both')
    # 如果首尾还有NaN，用后向/前向填充兜底
    data[numeric_cols] = data[numeric_cols].fillna(method='bfill').fillna(method='ffill')

    # 2. 核心特征工程（保留原有）
    noise_data = data[noise_columns]
    data['noise_mean'] = noise_data.mean(axis=1)
    data['noise_std'] = noise_data.std(axis=1)

    # 3. [新增] 强力特征：时序滞后与滑动窗口 (Lag & Rolling)
    # 注意：这会显著提升效果，但要求数据在时间上是连续的
    # 为了保持速度，只对 noise_mean 做时序特征
    data['noise_mean_lag1'] = data['noise_mean'].shift(1).fillna(method='bfill')  # 上一时刻的噪声均值
    data['noise_mean_roll3'] = data['noise_mean'].rolling(window=3, min_periods=1).mean()  # 最近3时刻的均值

    # 4. [新增] 交互特征：信噪比的代理变量
    # 假设 Error_T_SONIC 和 Error_CO2_density 有物理关联
    data['interact_err_ratio'] = data['Error_T_SONIC'] / (data['Error_CO2_density'] + 1e-6)

    # 5. 时间特征
    if TIME_COLUMN in data.columns:
        data['hour'] = data[TIME_COLUMN].dt.hour

    # 确定特征列
    added_feats = ['noise_mean', 'noise_std', 'noise_mean_lag1', 'noise_mean_roll3', 'interact_err_ratio']
    feature_cols = noise_columns + added_feats
    if TIME_COLUMN in data.columns:
        feature_cols += ['hour']

    # 6. 鲁棒缩放
    if is_train:
        scaler = RobustScaler()
        data[feature_cols] = scaler.fit_transform(data[feature_cols])
    else:
        data[feature_cols] = scaler.transform(data[feature_cols])

    return data, feature_cols, scaler


# 执行预处理
print("开始强化版数据预处理...")
train_processed, feature_cols, scaler = preprocess_data_enhanced(train_dataSet, is_train=True)
test_processed, _, _ = preprocess_data_enhanced(test_dataSet, is_train=False, scaler=scaler)

# ---------------------- 数据划分 ----------------------
X_train = train_processed[feature_cols]
y_train = train_processed[columns]
X_test = test_processed[feature_cols]
y_test = test_processed[columns]

print(f"特征数量: {len(feature_cols)} 列")
print(f"训练集：X{X_train.shape}")

# ---------------------- 模型优化：ExtraTrees ----------------------
# 相比RandomForest：
# 1. 训练速度更快 (Split更随机，计算量小)
# 2. 抗噪能力更强 (Variance更小)
model = ExtraTreesRegressor(
    n_estimators=300,  # 稍微增加树的数量，因为ET很快
    max_depth=20,  # 适当加深深度，捕捉更多细节
    min_samples_split=5,  # 允许更细的划分
    min_samples_leaf=2,
    max_features=0.7,  # 使用70%特征，增加随机性
    bootstrap=True,  # 开启Bootstrap以减少过拟合
    random_state=217,
    n_jobs=-1,
    verbose=0
)

# ---------------------- 训练与预测 ----------------------
print("\n开始训练 (ExtraTrees)...")
model.fit(X_train, y_train)
print("开始预测...")
y_predict = model.predict(X_test)


# ---------------------- 评估指标 ----------------------
def simple_metrics(y_true, y_pred):
    print("\n" + "=" * 50)
    print(">>> 评估报告 <<<")
    avg_mae = 0
    avg_mse = 0

    # 获取列名列表
    target_names = y_true.columns.tolist()

    print(f"{'特征名':<25} | {'MAE':<10} | {'MSE':<10}")
    print("-" * 50)

    for i, col in enumerate(target_names):
        mae = np.mean(np.abs(y_true.iloc[:, i] - y_pred[:, i]))
        mse = mean_squared_error(y_true.iloc[:, i], y_pred[:, i])
        avg_mae += mae
        avg_mse += mse
        print(f"{col:<25} | {mae:.4f}     | {mse:.4f}")

    avg_mae /= len(target_names)
    avg_rmse = np.sqrt(avg_mse / len(target_names))

    print("-" * 50)
    print(f"总体平均 MAE : {avg_mae:.5f} (越低越好)")
    print(f"总体平均 RMSE: {avg_rmse:.5f} (越低越好)")
    print("=" * 50)


simple_metrics(y_test, y_predict)

# ---------------------- 程序结束 ----------------------
end_time = time.time()
total_time = end_time - start_time
print(f"\n程序总耗时：{total_time:.2f}秒")


# ---------------------- [新增] 保存预测结果为比赛格式 ----------------------
# ==========================================
# 请将此代码段粘贴到 ExtraTrees 训练代码的最后面
# ==========================================

# 1. 定义保存路径 (请确保文件夹存在)
# 注意：这里文件名叫 result_ExtraTrees.csv，方便和之前的区分
output_csv_path = r'D:\ML期末代码比赛\004-比较代码\result_ExtraTrees.csv'

print(f"\n正在生成符合格式的预测文件: {output_csv_path} ...")

# 2. 格式化数据
# 将每一行的预测数值（列表）转换为一个用空格分隔的字符串
# 例如: [298.1, 0.5] -> "298.1 0.5"
formatted_rows = []
for row in y_predict:
    # map(str, row) 把数字转字符，" ".join 用空格连接
    row_str = " ".join(map(str, row))
    formatted_rows.append(row_str)

# 3. 创建 DataFrame
# 列名必须严格是 'Predicted_Value'
df_save = pd.DataFrame({'Predicted_Value': formatted_rows})

# 4. 保存文件 【关键点！】
# index=False 是为了不把行号(0,1,2...)写进去，避免出现 "Expected 1 fields, saw 2" 的报错
df_save.to_csv(output_csv_path, index=False)

print(f"文件保存成功！无索引，格式纯净。")
print(f"请现在去运行 evaluate_predictions.py 进行评估。")