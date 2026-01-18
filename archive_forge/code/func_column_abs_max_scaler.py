from typing import List, Tuple
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import AbsMax, Max, Mean, Min, Std
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def column_abs_max_scaler(s: pd.Series):
    s_abs_max = self.stats_[f'abs_max({s.name})']
    if s_abs_max == 0:
        s_abs_max = 1
    return s / s_abs_max