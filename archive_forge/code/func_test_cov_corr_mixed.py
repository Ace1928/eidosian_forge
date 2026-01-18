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
@pytest.mark.parametrize('numeric_only', [pytest.param(None, marks=pytest.mark.xfail(PANDAS_GE_200, reason='fails with non-numeric data')), pytest.param(True, marks=pytest.mark.skipif(not PANDAS_GE_150, reason='numeric_only not yet implemented')), pytest.param(False, marks=[pytest.mark.skipif(not PANDAS_GE_150, reason='numeric_only not yet implemented'), pytest.mark.xfail(PANDAS_GE_150, reason='fails with non-numeric data')])])
def test_cov_corr_mixed(numeric_only):
    size = 1000
    d = {'dates': pd.date_range('2015-01-01', periods=size, freq='1min'), 'unique_id': np.arange(0, size), 'ints': np.random.randint(0, size, size=size), 'floats': np.random.randn(size), 'bools': np.random.choice([0, 1], size=size), 'int_nans': np.random.choice([0, 1, np.nan], size=size), 'float_nans': np.random.choice([0.0, 1.0, np.nan], size=size), 'constant': 1, 'int_categorical': np.random.choice([10, 20, 30, 40, 50], size=size), 'categorical_binary': np.random.choice(['a', 'b'], size=size), 'categorical_nans': np.random.choice(['a', 'b', 'c'], size=size)}
    df = pd.DataFrame(d)
    df['hardbools'] = df['bools'] == 1
    df['categorical_nans'] = df['categorical_nans'].replace('c', np.nan)
    df['categorical_binary'] = df['categorical_binary'].astype('category')
    df['unique_id'] = df['unique_id'].astype(str)
    ddf = dd.from_pandas(df, npartitions=20)
    numeric_only_kwarg = {}
    if numeric_only is not None:
        numeric_only_kwarg = {'numeric_only': numeric_only}
    if not numeric_only_kwarg and PANDAS_GE_150 and (not PANDAS_GE_200):
        ctx = pytest.warns(FutureWarning, match='default value of numeric_only')
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        expected = df.corr(**numeric_only_kwarg)
    with ctx:
        result = ddf.corr(split_every=4, **numeric_only_kwarg)
    assert_eq(result, expected, check_divisions=False)
    with ctx:
        expected = df.cov(**numeric_only_kwarg)
    with ctx:
        result = ddf.cov(split_every=4, **numeric_only_kwarg)
    assert_eq(result, expected, check_divisions=False)