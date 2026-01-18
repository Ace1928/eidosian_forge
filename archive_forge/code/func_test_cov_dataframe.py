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
@pytest.mark.parametrize('numeric_only', [None, pytest.param(True, marks=pytest.mark.skipif(not PANDAS_GE_150, reason='numeric_only not yet implemented')), pytest.param(False, marks=pytest.mark.skipif(not PANDAS_GE_150, reason='numeric_only not yet implemented'))])
def test_cov_dataframe(numeric_only):
    df = _compat.makeMissingDataframe()
    ddf = dd.from_pandas(df, npartitions=6)
    numeric_only_kwarg = {}
    if numeric_only is not None:
        numeric_only_kwarg = {'numeric_only': numeric_only}
    res = ddf.cov(**numeric_only_kwarg)
    res2 = ddf.cov(**numeric_only_kwarg, split_every=2)
    res3 = ddf.cov(10, **numeric_only_kwarg)
    res4 = ddf.cov(10, **numeric_only_kwarg, split_every=2)
    sol = df.cov(**numeric_only_kwarg)
    sol2 = df.cov(10, **numeric_only_kwarg)
    assert_eq(res, sol)
    assert_eq(res2, sol)
    assert_eq(res3, sol2)
    assert_eq(res4, sol2)
    assert res._name == ddf.cov(**numeric_only_kwarg)._name
    assert res._name != res2._name
    assert res3._name != res4._name
    assert res._name != res3._name