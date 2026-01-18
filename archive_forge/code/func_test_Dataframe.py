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
def test_Dataframe():
    expected = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10], index=[0, 1, 3, 5, 6, 8, 9, 9, 9], name='a')
    assert_eq(d['a'] + 1, expected)
    tm.assert_index_equal(d.columns, pd.Index(['a', 'b']))
    assert_eq(d[d['b'] > 2], full[full['b'] > 2])
    assert_eq(d[['a', 'b']], full[['a', 'b']])
    assert_eq(d.a, full.a)
    assert d.b.mean().compute() == full.b.mean()
    assert np.allclose(d.b.var().compute(), full.b.var())
    assert np.allclose(d.b.std().compute(), full.b.std())
    assert d.index._name == d.index._name
    assert repr(d)