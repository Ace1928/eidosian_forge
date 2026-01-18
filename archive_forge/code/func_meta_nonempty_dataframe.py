from __future__ import annotations
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar, union_categoricals
from dask.array.core import Array
from dask.array.dispatch import percentile_lookup
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
from dask.dataframe._compat import PANDAS_GE_220, is_any_real_numeric_dtype
from dask.dataframe.core import DataFrame, Index, Scalar, Series, _Frame
from dask.dataframe.dispatch import (
from dask.dataframe.extensions import make_array_nonempty, make_scalar
from dask.dataframe.utils import (
from dask.sizeof import SimpleSizeof, sizeof
from dask.utils import is_arraylike, is_series_like, typename
@meta_nonempty.register(pd.DataFrame)
def meta_nonempty_dataframe(x):
    idx = meta_nonempty(x.index)
    dt_s_dict = dict()
    data = dict()
    for i in range(len(x.columns)):
        series = x.iloc[:, i]
        dt = series.dtype
        if dt not in dt_s_dict:
            dt_s_dict[dt] = _nonempty_series(x.iloc[:, i], idx=idx)
        data[i] = dt_s_dict[dt]
    res = pd.DataFrame(data, index=idx, columns=np.arange(len(x.columns)))
    res.columns = x.columns
    res.attrs = x.attrs
    return res