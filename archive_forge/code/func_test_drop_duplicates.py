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
@pytest.mark.parametrize('shuffle_method', [None, pytest.param(True, marks=pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not supported'))])
def test_drop_duplicates(shuffle_method):
    res = d.drop_duplicates()
    res2 = d.drop_duplicates(split_every=2, shuffle_method=shuffle_method)
    sol = full.drop_duplicates()
    assert_eq(res, sol)
    assert_eq(res2, sol)
    assert res._name != res2._name
    res = d.a.drop_duplicates()
    res2 = d.a.drop_duplicates(split_every=2, shuffle_method=shuffle_method)
    sol = full.a.drop_duplicates()
    assert_eq(res, sol)
    assert_eq(res2, sol)
    assert res._name != res2._name
    res = d.index.drop_duplicates()
    res2 = d.index.drop_duplicates(split_every=2, shuffle_method=shuffle_method)
    sol = full.index.drop_duplicates()
    if DASK_EXPR_ENABLED:
        assert_eq(res.compute().sort_values(), sol)
        assert_eq(res2.compute().sort_values(), sol)
    else:
        assert_eq(res, sol)
        assert_eq(res2, sol)
    _d = d.clear_divisions()
    res = _d.index.drop_duplicates()
    res2 = _d.index.drop_duplicates(split_every=2, shuffle_method=shuffle_method)
    sol = full.index.drop_duplicates()
    if DASK_EXPR_ENABLED:
        assert_eq(res.compute().sort_values(), sol)
        assert_eq(res2.compute().sort_values(), sol)
    else:
        assert_eq(res, sol)
        assert_eq(res2, sol)
    assert res._name != res2._name
    with pytest.raises(NotImplementedError):
        d.drop_duplicates(keep=False)