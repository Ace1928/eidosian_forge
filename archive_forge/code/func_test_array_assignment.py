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
def test_array_assignment():
    df = pd.DataFrame({'x': np.random.normal(size=50), 'y': np.random.normal(size=50)})
    ddf = dd.from_pandas(df, npartitions=2)
    orig = ddf.copy()
    arr = np.array(np.random.normal(size=50))
    darr = da.from_array(arr, chunks=25)
    df['z'] = arr
    ddf['z'] = darr
    assert_eq(df, ddf)
    assert 'z' not in orig.columns
    arr = np.array(np.random.normal(size=(50, 50)))
    darr = da.from_array(arr, chunks=25)
    msg = 'Array assignment only supports 1-D arrays'
    with pytest.raises(ValueError, match=msg):
        ddf['z'] = darr
    arr = np.array(np.random.normal(size=50))
    darr = da.from_array(arr, chunks=10)
    msg = 'Number of partitions do not match'
    with pytest.raises(ValueError, match=msg):
        ddf['z'] = darr