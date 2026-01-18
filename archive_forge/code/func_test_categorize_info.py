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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not compatible')
def test_categorize_info():
    from io import StringIO
    pandas_format._put_lines = put_lines
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': pd.Series(list('aabc')), 'z': pd.Series(list('aabc'))}, index=[0, 1, 2, 3])

    def myfunc(bounds):
        start, stop = bounds
        return df.iloc[start:stop]
    ddf = dd.from_map(myfunc, [(0, 1), (1, 2), (2, 4)], divisions=[0, 1, 2, 3]).categorize(['y'])
    buf = StringIO()
    ddf.info(buf=buf, verbose=True)
    string_dtype = 'object' if get_string_dtype() is object else 'string'
    memory_usage = float(ddf.memory_usage().sum().compute())
    if pyarrow_strings_enabled():
        dtypes = f'category(1), int64(1), {string_dtype}(1)'
    else:
        dtypes = f'category(1), {string_dtype}(1), int64(1)'
    expected = dedent(f"        <class 'dask.dataframe.core.DataFrame'>\n        {type(ddf._meta.index).__name__}: 4 entries, 0 to 3\n        Data columns (total 3 columns):\n         #   Column  Non-Null Count  Dtype\n        ---  ------  --------------  -----\n         0   x       4 non-null      int64\n         1   y       4 non-null      category\n         2   z       4 non-null      {string_dtype}\n        dtypes: {dtypes}\n        memory usage: {memory_usage} bytes\n        ")
    assert buf.getvalue() == expected