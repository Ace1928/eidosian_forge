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
@pytest.mark.parametrize('columns', [('x', 'y'), pytest.param(('x', 'x'), marks=pytest.mark.xfail(DASK_EXPR_ENABLED, reason='duplicated columns')), pytest.param(pd.MultiIndex.from_tuples([('x', 1), ('x', 2)], names=('letter', 'number')), marks=pytest.mark.skipif(DASK_EXPR_ENABLED, reason='Midx columns'))])
def test_dataframe_items(columns):
    df = pd.DataFrame([[1, 10], [2, 20], [3, 30], [4, 40]], columns=columns)
    ddf = dd.from_pandas(df, npartitions=2)
    for a, b in zip(df.items(), ddf.items()):
        assert a[0] == b[0]
        assert_eq(a[1], b[1].compute())