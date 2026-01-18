import math
import string
from datetime import datetime, timedelta
from functools import lru_cache
from itertools import cycle
import numpy as np
import pandas as pd
from .utils import find_package_file
def generate_random_series(rows, type):
    if type == 'bool':
        return pd.Series(np.random.binomial(n=1, p=0.5, size=rows), dtype=bool)
    if type == 'boolean':
        x = generate_random_series(rows, 'bool').astype(type)
        x.loc[np.random.binomial(n=1, p=0.1, size=rows) == 0] = pd.NA
        return x
    if type == 'int':
        return pd.Series(np.random.geometric(p=0.1, size=rows), dtype=int)
    if type == 'Int64':
        x = generate_random_series(rows, 'int').astype(type)
        if PANDAS_VERSION_MAJOR >= 1:
            x.loc[np.random.binomial(n=1, p=0.1, size=rows) == 0] = pd.NA
        return x
    if type == 'float':
        x = pd.Series(np.random.normal(size=rows), dtype=float)
        x.loc[np.random.binomial(n=1, p=0.05, size=rows) == 0] = float('nan')
        x.loc[np.random.binomial(n=1, p=0.05, size=rows) == 0] = float('inf')
        x.loc[np.random.binomial(n=1, p=0.05, size=rows) == 0] = float('-inf')
        return x
    if type == 'str':
        return get_countries()['region'].sample(n=rows, replace=True)
    if type == 'categories':
        x = generate_random_series(rows, 'str')
        return pd.Series(x, dtype='category')
    if type == 'date':
        x = generate_date_series().sample(rows, replace=True)
        x.loc[np.random.binomial(n=1, p=0.1, size=rows) == 0] = pd.NaT
        return x
    if type == 'datetime':
        x = generate_random_series(rows, 'date') + np.random.uniform(0, 1, rows) * pd.Timedelta(1, unit='D')
        return x
    if type == 'timedelta':
        x = generate_random_series(rows, 'datetime').sample(frac=1)
        return x.diff()
    raise NotImplementedError(type)