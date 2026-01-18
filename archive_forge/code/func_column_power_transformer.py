from typing import List
import numpy as np
import pandas as pd
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def column_power_transformer(s: pd.Series):
    if self.method == 'yeo-johnson':
        result = np.zeros_like(s, dtype=np.float64)
        pos = s >= 0
        if self.power != 0:
            result[pos] = (np.power(s[pos] + 1, self.power) - 1) / self.power
        else:
            result[pos] = np.log(s[pos] + 1)
        if self.power != 2:
            result[~pos] = -(np.power(-s[~pos] + 1, 2 - self.power) - 1) / (2 - self.power)
        else:
            result[~pos] = -np.log(-s[~pos] + 1)
        return result
    elif self.power != 0:
        return (np.power(s, self.power) - 1) / self.power
    else:
        return np.log(s)