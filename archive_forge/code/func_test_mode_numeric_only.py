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
def test_mode_numeric_only():
    df = pd.DataFrame({'int': [1, 2, 3, 4, 5, 6, 7, 8], 'float': [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0], 'dt': [pd.NaT] + [datetime(2010, i, 1) for i in range(1, 8)], 'timedelta': pd.to_timedelta([1, 2, 3, 4, 5, 6, 7, np.nan])})
    ddf = dd.from_pandas(df, npartitions=2)
    assert_eq(ddf.mode(numeric_only=False), df.mode(numeric_only=False))
    assert_eq(ddf.mode(), df.mode())
    assert_eq(ddf.mode(numeric_only=True), df.mode(numeric_only=True))