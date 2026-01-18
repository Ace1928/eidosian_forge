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
@meta_nonempty.register(object)
def meta_nonempty_object(x):
    """Create a nonempty pandas object from the given metadata.

    Returns a pandas DataFrame, Series, or Index that contains two rows
    of fake data.
    """
    if is_scalar(x):
        return _nonempty_scalar(x)
    else:
        raise TypeError(f'Expected Pandas-like Index, Series, DataFrame, or scalar, got {typename(type(x))}')