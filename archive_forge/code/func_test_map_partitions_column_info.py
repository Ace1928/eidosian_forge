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
def test_map_partitions_column_info():
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [5, 6, 7, 8]})
    a = dd.from_pandas(df, npartitions=2)
    b = dd.map_partitions(lambda x: x, a, meta=a)
    tm.assert_index_equal(b.columns, a.columns)
    assert_eq(df, b)
    b = dd.map_partitions(lambda x: x, a.x, meta=a.x)
    assert b.name == a.x.name
    assert_eq(df.x, b)
    b = dd.map_partitions(lambda x: x, a.x, meta=a.x)
    assert b.name == a.x.name
    assert_eq(df.x, b)
    b = dd.map_partitions(lambda df: df.x + df.y, a)
    assert isinstance(b, dd.Series)
    assert b.dtype == 'i8'
    b = dd.map_partitions(lambda df: df.x + 1, a, meta=('x', 'i8'))
    assert isinstance(b, dd.Series)
    assert b.name == 'x'
    assert b.dtype == 'i8'