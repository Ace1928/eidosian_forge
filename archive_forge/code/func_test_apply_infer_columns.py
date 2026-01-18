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
def test_apply_infer_columns():
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [10, 20, 30, 40]})
    ddf = dd.from_pandas(df, npartitions=2)

    def return_df(x):
        return pd.Series([x.sum(), x.mean()], index=['sum', 'mean'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        result = ddf.apply(return_df, axis=1)
    assert isinstance(result, dd.DataFrame)
    tm.assert_index_equal(result.columns, pd.Index(['sum', 'mean']))
    assert_eq(result, df.apply(return_df, axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        result = ddf.apply(lambda x: 1, axis=1)
    assert isinstance(result, dd.Series)
    assert result.name is None
    assert_eq(result, df.apply(lambda x: 1, axis=1))

    def return_df2(x):
        return pd.Series([x * 2, x * 3], index=['x2', 'x3'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        result = ddf.x.apply(return_df2)
    assert isinstance(result, dd.DataFrame)
    tm.assert_index_equal(result.columns, pd.Index(['x2', 'x3']))
    assert_eq(result, df.x.apply(return_df2))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        result = ddf.x.apply(lambda x: 1)
    assert isinstance(result, dd.Series)
    assert result.name == 'x'
    assert_eq(result, df.x.apply(lambda x: 1))