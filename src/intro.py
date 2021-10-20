"""
This module:

- import all datasets
- concatenate train and test data
- convert timestamp columns to datetime type
- select columns of interest
- interpolate missing numeric values
- merge failure and signals data
- add wind segment variable
- make separate data frames for each wind turbine (T01, T06, T07, T09, T11)

"""

import pandas as pd
import numpy as np

power_curve = pd.read_csv("../res/Power_curve.csv")

failures = pd.read_csv("../res/wind-farm-1-failures-training.csv", sep=";")
logs_train = pd.read_csv("../res/wind-farm-1-logs-training.csv", sep=";")
metmast_train = pd.read_csv("../res/wind-farm-1-metmast-training.csv", sep=";")
signals_train = pd.read_csv("../res/wind-farm-1-signals-training.csv", sep=";")

logs_test = pd.read_csv("../res/wind-farm-1-logs-testing.csv", sep=";")
metmast_test = pd.read_csv("../res/wind-farm-1-metmast-testing.csv", sep=";")
signals_test = pd.read_csv("../res/wind-farm-1-signals-testing.csv", sep=";")

logs = pd.concat([logs_train, logs_test])
metmast = pd.concat([metmast_train, metmast_test])
signals = pd.concat([signals_train, signals_test])

power_curve.columns = ["wind_speed", "power"]

failures.Timestamp = pd.to_datetime(failures.Timestamp)
logs.TimeDetected = pd.to_datetime(logs.TimeDetected)
logs.TimeReset = pd.to_datetime(logs.TimeReset)
metmast.Timestamp = pd.to_datetime(metmast.Timestamp)
signals.Timestamp = pd.to_datetime(signals.Timestamp)

cols = ["Turbine_ID", "Timestamp", "Grd_Prod_Pwr_Avg", "Amb_WindSpeed_Avg", "Amb_Temp_Avg", "Nac_Temp_Avg",
            "Rtr_RPM_Avg", "Gear_Bear_Temp_Avg", "Gen_RPM_Avg", "Gen_Bear2_Temp_Avg", "Gen_Bear_Temp_Avg",
            "Prod_LatestAvg_TotActPwr", "Hyd_Oil_Temp_Avg", "Gear_Oil_Temp_Avg"]

signals.drop(columns=signals.columns.difference(cols), inplace=True)

signals.Gen_Bear_Temp_Avg = signals.Gen_Bear_Temp_Avg.interpolate() # linear interpolation of missing values

failures.Timestamp = failures.Timestamp.dt.round(freq="10min")
signals = signals.merge(failures, on=["Turbine_ID", "Timestamp"], how="left")

failure_idx = list(signals[signals.Component.notna()].index)
signals["Failure"] = 0
signals.loc[failure_idx, "Failure"] = 1

def wind_segmentation(x):
    if x < 4:
        return 0
    elif 4 <= x < 4.5:
        return 1
    elif 4.5 <= x < 5:
        return 2
    elif 5 <= x < 5.5:
        return 3
    elif 5.5 <= x < 6:
        return 4
    elif 6 <= x < 6.5:
        return 5
    elif 6.5 <= x < 7:
        return 6
    elif 7 <= x < 7.5:
        return 7
    elif 7.5 <= x < 8:
        return 8
    elif 8 <= x < 8.5:
        return 9
    elif 8.5 <= x < 9:
        return 10
    else:
        return 11

signals["Wind_Segment_Code"] = signals.Amb_WindSpeed_Avg.apply(wind_segmentation)

signals_T01 = signals[signals.Turbine_ID == "T01"].reset_index(drop=True)
signals_T06 = signals[signals.Turbine_ID == "T06"].reset_index(drop=True)
signals_T07 = signals[signals.Turbine_ID == "T07"].reset_index(drop=True)
signals_T09 = signals[signals.Turbine_ID == "T09"].reset_index(drop=True)
signals_T11 = signals[signals.Turbine_ID == "T11"].reset_index(drop=True)