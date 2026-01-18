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
def test_deterministic_apply_concat_apply_names():
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [5, 6, 7, 8]})
    a = dd.from_pandas(df, npartitions=2)
    assert sorted(a.x.nlargest(2).dask) == sorted(a.x.nlargest(2).dask)
    assert sorted(a.x.nlargest(2).dask) != sorted(a.x.nlargest(3).dask)
    assert sorted(a.x.drop_duplicates().dask) == sorted(a.x.drop_duplicates().dask)
    assert sorted(a.groupby('x').y.mean().dask) == sorted(a.groupby('x').y.mean().dask)
    f = lambda a: a.nlargest(5)
    f2 = lambda a: a.nlargest(3)
    if not DASK_EXPR_ENABLED:
        assert sorted(aca(a.x, f, f, a.x._meta).dask) != sorted(aca(a.x, f2, f2, a.x._meta).dask)
        assert sorted(aca(a.x, f, f, a.x._meta).dask) == sorted(aca(a.x, f, f, a.x._meta).dask)

        def chunk(x, c_key=0, both_key=0):
            return x.sum() + c_key + both_key

        def agg(x, a_key=0, both_key=0):
            return pd.Series(x).sum() + a_key + both_key
        c_key = 2
        a_key = 3
        both_key = 4
        res = aca(a.x, chunk=chunk, aggregate=agg, chunk_kwargs={'c_key': c_key}, aggregate_kwargs={'a_key': a_key}, both_key=both_key)
        assert sorted(res.dask) == sorted(aca(a.x, chunk=chunk, aggregate=agg, chunk_kwargs={'c_key': c_key}, aggregate_kwargs={'a_key': a_key}, both_key=both_key).dask)
        assert sorted(res.dask) != sorted(aca(a.x, chunk=chunk, aggregate=agg, chunk_kwargs={'c_key': c_key}, aggregate_kwargs={'a_key': a_key}, both_key=0).dask)
        assert_eq(res, df.x.sum() + 2 * (c_key + both_key) + a_key + both_key)