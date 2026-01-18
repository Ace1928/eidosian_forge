from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def group_iter(self):
    for low, upp in self.groupidx:
        yield slice(low, upp)