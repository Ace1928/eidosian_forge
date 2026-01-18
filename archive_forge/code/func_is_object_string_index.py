from __future__ import annotations
from functools import partial
import pandas as pd
from packaging.version import Version
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like
def is_object_string_index(x):
    if isinstance(x, pd.MultiIndex):
        return any((is_object_string_index(level) for level in x.levels))
    return isinstance(x, pd.Index) and is_object_string_dtype(x.dtype)