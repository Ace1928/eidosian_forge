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
def test_map_partitions_propagates_index_metadata():
    index = pd.Series(list('abcde'), name='myindex')
    df = pd.DataFrame({'A': np.arange(5, dtype=np.int32), 'B': np.arange(10, 15, dtype=np.int32)}, index=index)
    ddf = dd.from_pandas(df, npartitions=2)
    res = ddf.map_partitions(lambda df: df.assign(C=df.A + df.B), meta=[('A', 'i4'), ('B', 'i4'), ('C', 'i4')])
    sol = df.assign(C=df.A + df.B)
    assert_eq(res, sol)
    res = ddf.map_partitions(lambda df: df.rename_axis('newindex'))
    sol = df.rename_axis('newindex')
    assert_eq(res, sol)