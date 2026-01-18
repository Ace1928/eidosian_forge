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
def test_dot_nan():
    s1 = pd.Series([1, 2, 3, 4])
    dask_s1 = dd.from_pandas(s1, npartitions=1)
    s2 = pd.Series([np.nan, np.nan, np.nan, np.nan])
    dask_s2 = dd.from_pandas(s2, npartitions=1)
    df = pd.DataFrame({'one': s1, 'two': s2})
    dask_df = dd.from_pandas(df, npartitions=1)
    assert_eq(s1.dot(s2), dask_s1.dot(dask_s2))
    assert_eq(s2.dot(df), dask_s2.dot(dask_df))