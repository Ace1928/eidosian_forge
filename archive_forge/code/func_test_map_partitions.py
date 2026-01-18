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
def test_map_partitions():
    assert_eq(d.map_partitions(lambda df: df, meta=d), full)
    assert_eq(d.map_partitions(lambda df: df), full)
    result = d.map_partitions(lambda df: df.sum(axis=1))
    if not DASK_EXPR_ENABLED:
        layer = hlg_layer(result.dask, 'lambda-')
        assert not layer.is_materialized(), layer
    assert_eq(result, full.sum(axis=1))
    assert_eq(d.map_partitions(lambda df: 1), pd.Series([1, 1, 1], dtype=np.int64), check_divisions=False)
    if not DASK_EXPR_ENABLED:
        x = Scalar({('x', 0): 1}, 'x', int)
        result = dd.map_partitions(lambda x: 2, x)
        assert result.dtype in (np.int32, np.int64) and result.compute() == 2
        result = dd.map_partitions(lambda x: 4.0, x)
        assert result.dtype == np.float64 and result.compute() == 4.0