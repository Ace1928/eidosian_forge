from __future__ import annotations
import warnings
from functools import partial
import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype
from pandas.errors import PerformanceWarning
from tlz import partition
from dask.dataframe._compat import (
from dask.dataframe.dispatch import (  # noqa: F401
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like
from dask.utils import _deprecated_kwarg
def describe_nonnumeric_aggregate(stats, name):
    args_len = len(stats)
    is_datetime_column = args_len == 5
    is_categorical_column = args_len == 3
    assert is_datetime_column or is_categorical_column
    if is_categorical_column:
        nunique, count, top_freq = stats
    else:
        nunique, count, top_freq, min_ts, max_ts = stats
    if len(top_freq) == 0:
        data = [0, 0]
        index = ['count', 'unique']
        dtype = None
        data.extend([np.nan, np.nan])
        index.extend(['top', 'freq'])
        dtype = object
        result = pd.Series(data, index=index, dtype=dtype, name=name)
        return result
    top = top_freq.index[0]
    freq = top_freq.iloc[0]
    index = ['unique', 'count', 'top', 'freq']
    values = [nunique, count]
    if is_datetime_column:
        tz = top.tz
        top = pd.Timestamp(top)
        if top.tzinfo is not None and tz is not None:
            top = top.tz_convert(tz)
        else:
            top = top.tz_localize(tz)
        first = pd.Timestamp(min_ts, tz=tz)
        last = pd.Timestamp(max_ts, tz=tz)
        index.extend(['first', 'last'])
        values.extend([top, freq, first, last])
    else:
        values.extend([top, freq])
    return pd.Series(values, index=index, name=name)