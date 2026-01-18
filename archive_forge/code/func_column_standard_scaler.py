from typing import List, Tuple
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import AbsMax, Max, Mean, Min, Std
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def column_standard_scaler(s: pd.Series):
    s_mean = self.stats_[f'mean({s.name})']
    s_std = self.stats_[f'std({s.name})']
    if s_std == 0:
        s_std = 1
    return (s - s_mean) / s_std