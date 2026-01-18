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
@pytest.mark.parametrize('index, left, right', [(range(10), 0, 9), (range(10), -1, None), (range(10), None, 10), ([-1, 0, 2, 1], None, None), ([-1, 0, 2, 1], -1, None), ([-1, 0, 2, 1], None, 2), ([-1, 0, 2, 1], -2, 3), (pd.date_range('2017', periods=10), None, None), (pd.date_range('2017', periods=10), pd.Timestamp('2017'), None), (pd.date_range('2017', periods=10), None, pd.Timestamp('2017-01-10')), (pd.date_range('2017', periods=10), pd.Timestamp('2016'), None), (pd.date_range('2017', periods=10), None, pd.Timestamp('2018'))])
def test_boundary_slice_same(index, left, right):
    df = pd.DataFrame({'A': range(len(index))}, index=index)
    result = methods.boundary_slice(df, left, right)
    tm.assert_frame_equal(result, df)