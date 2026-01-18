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
@pytest.mark.parametrize('method,test_values', [pytest.param('tdigest', (6, 10), marks=pytest.mark.skipif(not crick, reason='Requires crick')), ('dask', (4, 20))])
def test_describe_numeric(method, test_values):
    s = pd.Series(list(range(test_values[1])) * test_values[0])
    df = pd.DataFrame({'a': list(range(test_values[1])) * test_values[0], 'b': list(range(test_values[0])) * test_values[1]})
    ds = dd.from_pandas(s, test_values[0])
    ddf = dd.from_pandas(df, test_values[0])
    test_quantiles = [0.25, 0.75]
    assert_eq(df.describe(), ddf.describe(percentiles_method=method))
    assert_eq(s.describe(), ds.describe(percentiles_method=method))
    assert_eq(df.describe(percentiles=test_quantiles), ddf.describe(percentiles=test_quantiles, percentiles_method=method))
    assert_eq(s.describe(), ds.describe(split_every=2, percentiles_method=method))
    assert_eq(df.describe(), ddf.describe(split_every=2, percentiles_method=method))
    df = pd.DataFrame({'a': list(range(test_values[1])) * test_values[0], 'b': list(range(test_values[0])) * test_values[1], 'c': list('abcdef'[:test_values[0]]) * test_values[1]})
    ddf = dd.from_pandas(df, test_values[0])
    assert_eq(df.describe(), ddf.describe(percentiles_method=method))
    assert_eq(df.describe(), ddf.describe(split_every=2, percentiles_method=method))