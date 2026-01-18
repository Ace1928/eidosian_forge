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
@pytest.mark.xfail(PANDAS_GE_210, reason='https://github.com/dask/dask/issues/10931')
@pytest.mark.parametrize('dtype_lhs,dtype_rhs', [('f8', 'i8'), ('f8', 'f4'), ('datetime64[s]', 'datetime64[ns]')])
def test_combine_first_all_nans(dtype_lhs, dtype_rhs):
    """If you call s1.combine_first(s2), where s1 is pandas.Series of all NaNs and s2 is
    a pandas.Series of non-floats, the dtype becomes that of s2. Starting with pandas
    2.1, this comes with a deprecation warning.
    Test behaviour when either a whole dask series or just a chunk is full of NaNs.
    """
    if PANDAS_GE_210:
        ctx = pytest.warns(FutureWarning, match='The behavior of array concatenation with empty entries is deprecated')
    else:
        ctx = contextlib.nullcontext()
    s1 = pd.Series([np.nan, np.nan], dtype=dtype_lhs)
    s2 = pd.Series([np.nan, 1.0], dtype=dtype_lhs)
    s3 = pd.Series([1, 2], dtype=dtype_rhs)
    ds1 = dd.from_pandas(s1, npartitions=2)
    ds2 = dd.from_pandas(s2, npartitions=2)
    ds3 = dd.from_pandas(s3, npartitions=2)
    with ctx:
        s13 = s1.combine_first(s3)
    with ctx:
        assert_eq(ds1.combine_first(ds3), s13)
    assert_eq(ds2.combine_first(ds3), s2.combine_first(s3))