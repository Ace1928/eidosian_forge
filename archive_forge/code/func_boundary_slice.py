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
def boundary_slice(df, start, stop, right_boundary=True, left_boundary=True, kind=None):
    """Index slice start/stop. Can switch include/exclude boundaries.

    Examples
    --------
    >>> df = pd.DataFrame({'x': [10, 20, 30, 40, 50]}, index=[1, 2, 2, 3, 4])
    >>> boundary_slice(df, 2, None)
        x
    2  20
    2  30
    3  40
    4  50
    >>> boundary_slice(df, 1, 3)
        x
    1  10
    2  20
    2  30
    3  40
    >>> boundary_slice(df, 1, 3, right_boundary=False)
        x
    1  10
    2  20
    2  30

    Empty input DataFrames are returned

    >>> df_empty = pd.DataFrame()
    >>> boundary_slice(df_empty, 1, 3)
    Empty DataFrame
    Columns: []
    Index: []
    """
    if len(df.index) == 0:
        return df
    if PANDAS_GE_131:
        if kind is not None:
            warnings.warn('The `kind` argument is no longer used/supported. It will be dropped in a future release.', category=FutureWarning)
        kind_opts = {}
        kind = 'loc'
    else:
        kind = kind or 'loc'
        kind_opts = {'kind': kind}
    if kind == 'loc' and (not df.index.is_monotonic_increasing):
        if start is not None:
            if left_boundary:
                df = df[df.index >= start]
            else:
                df = df[df.index > start]
        if stop is not None:
            if right_boundary:
                df = df[df.index <= stop]
            else:
                df = df[df.index < stop]
        return df
    result = getattr(df, kind)[start:stop]
    if not right_boundary and stop is not None:
        right_index = result.index.get_slice_bound(stop, 'left', **kind_opts)
        result = result.iloc[:right_index]
    if not left_boundary and start is not None:
        left_index = result.index.get_slice_bound(start, 'right', **kind_opts)
        result = result.iloc[left_index:]
    return result