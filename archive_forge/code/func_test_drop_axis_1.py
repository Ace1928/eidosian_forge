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
def test_drop_axis_1():
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [5, 6, 7, 8], 'z': [9, 10, 11, 12]})
    ddf = dd.from_pandas(df, npartitions=2)
    assert_eq(ddf.drop('y', axis=1), df.drop('y', axis=1))
    assert_eq(ddf.drop(['y', 'z'], axis=1), df.drop(['y', 'z'], axis=1))
    with pytest.raises((ValueError, KeyError)):
        ddf.drop(['a', 'x'], axis=1)
    assert_eq(ddf.drop(['a', 'x'], axis=1, errors='ignore'), df.drop(['a', 'x'], axis=1, errors='ignore'))
    assert_eq(ddf.drop(columns=['y', 'z']), df.drop(columns=['y', 'z']))