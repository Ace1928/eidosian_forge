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
def test_groupby_callable():
    a = pd.DataFrame({'x': [1, 2, 3, None], 'y': [10, 20, 30, 40]}, index=[1, 2, 3, 4])
    b = dd.from_pandas(a, 2)

    def iseven(x):
        return x % 2 == 0
    assert_eq(a.groupby(iseven).y.sum(), b.groupby(iseven).y.sum())
    assert_eq(a.y.groupby(iseven).sum(), b.y.groupby(iseven).sum())