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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='deprecated in pandas')
@pytest.mark.parametrize('method', ['first', 'last'])
def test_first_and_last(method):
    f = lambda x, offset: getattr(x, method)(offset)
    freqs = ['12h', 'D']
    offsets = ['0d', '100h', '20d', '20B', '3W', '3M', '400d', '13M']
    for freq in freqs:
        index = pd.date_range('1/1/2000', '1/1/2001', freq=freq)[::4]
        df = pd.DataFrame(np.random.random((len(index), 4)), index=index, columns=['A', 'B', 'C', 'D'])
        ddf = dd.from_pandas(df, npartitions=10)
        for offset in offsets:
            with _check_warning(PANDAS_GE_210, FutureWarning, method):
                expected = f(df, offset)
            ctx = pytest.warns(FutureWarning, match='Will be removed in a future version.') if not PANDAS_GE_210 else contextlib.nullcontext()
            with _check_warning(PANDAS_GE_210, FutureWarning, method), ctx:
                actual = f(ddf, offset)
            assert_eq(actual, expected)
            with _check_warning(PANDAS_GE_210, FutureWarning, method):
                expected = f(df.A, offset)
            with _check_warning(PANDAS_GE_210, FutureWarning, method), ctx:
                actual = f(ddf.A, offset)
            assert_eq(actual, expected)