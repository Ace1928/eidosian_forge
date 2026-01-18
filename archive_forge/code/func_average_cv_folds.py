from typing import Dict, List
import sklearn.datasets
import sklearn.metrics
import os
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
def average_cv_folds(results_dict: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: np.mean(v) for k, v in results_dict.items()}