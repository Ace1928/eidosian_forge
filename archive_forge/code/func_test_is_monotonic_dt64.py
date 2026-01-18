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
def test_is_monotonic_dt64():
    s = pd.Series(pd.date_range('20130101', periods=10))
    ds = dd.from_pandas(s, npartitions=5)
    assert_eq(s.is_monotonic_increasing, ds.is_monotonic_increasing)
    assert_eq(s.is_monotonic_decreasing, ds.is_monotonic_decreasing)
    s_2 = pd.Series(list(reversed(s)))
    ds_2 = dd.from_pandas(s_2, npartitions=5)
    assert_eq(s_2.is_monotonic_increasing, ds_2.is_monotonic_increasing)
    assert_eq(s_2.is_monotonic_decreasing, ds_2.is_monotonic_decreasing)