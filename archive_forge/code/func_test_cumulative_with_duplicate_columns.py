from __future__ import annotations
import contextlib
import decimal
import warnings
import weakref
import xml.etree.ElementTree
from datetime import datetime, timedelta
from itertools import product
from operator import add
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from pandas.errors import PerformanceWarning
from pandas.io.formats import format as pandas_format
import dask
import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby
from dask import delayed
from dask.base import compute_as_if_collection
from dask.blockwise import fuse_roots
from dask.dataframe import _compat, methods
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import (
from dask.dataframe.utils import (
from dask.datasets import timeseries
from dask.utils import M, is_dataframe_like, is_series_like, put_lines
from dask.utils_test import _check_warning, hlg_layer
def test_cumulative_with_duplicate_columns():
    df = pd.DataFrame(np.random.randn(100, 3), columns=list('abb'))
    ddf = dd.from_pandas(df, 3)
    assert_eq(df.cumsum(), ddf.cumsum())
    assert_eq(df.cummin(), ddf.cummin())
    assert_eq(df.cummax(), ddf.cummax())
    assert_eq(df.cumprod(), ddf.cumprod())
    assert_eq(df.cumsum(skipna=False), ddf.cumsum(skipna=False))
    assert_eq(df.cummin(skipna=False), ddf.cummin(skipna=False))
    assert_eq(df.cummax(skipna=False), ddf.cummax(skipna=False))
    assert_eq(df.cumprod(skipna=False), ddf.cumprod(skipna=False))
    assert_eq(df.cumsum(axis=1), ddf.cumsum(axis=1))
    assert_eq(df.cummin(axis=1), ddf.cummin(axis=1))
    assert_eq(df.cummax(axis=1), ddf.cummax(axis=1))
    assert_eq(df.cumprod(axis=1), ddf.cumprod(axis=1))
    assert_eq(df.cumsum(axis=1, skipna=False), ddf.cumsum(axis=1, skipna=False))
    assert_eq(df.cummin(axis=1, skipna=False), ddf.cummin(axis=1, skipna=False))
    assert_eq(df.cummax(axis=1, skipna=False), ddf.cummax(axis=1, skipna=False))
    assert_eq(df.cumprod(axis=1, skipna=False), ddf.cumprod(axis=1, skipna=False))