from typing import List, Tuple
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import AbsMax, Max, Mean, Min, Std
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def column_robust_scaler(s: pd.Series):
    s_low_q = self.stats_[f'low_quantile({s.name})']
    s_median = self.stats_[f'median({s.name})']
    s_high_q = self.stats_[f'high_quantile({s.name})']
    diff = s_high_q - s_low_q
    if diff == 0:
        return np.zeros_like(s)
    return (s - s_median) / diff