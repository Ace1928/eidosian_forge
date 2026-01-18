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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='Not public')
def test_aca_meta_infer():
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [5, 6, 7, 8]})
    ddf = dd.from_pandas(df, npartitions=2)

    def chunk(x, y, constant=1.0):
        return (x + y + constant).head()

    def agg(x):
        return x.head()
    res = aca([ddf, 2.0], chunk=chunk, aggregate=agg, chunk_kwargs=dict(constant=2.0))
    sol = (df + 2.0 + 2.0).head()
    assert_eq(res, sol)
    res = aca([ddf.x], chunk=lambda x: pd.Series([x.sum()]), aggregate=lambda x: x.sum())
    assert isinstance(res, Scalar)
    assert res.compute() == df.x.sum()