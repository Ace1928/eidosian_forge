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
@pytest.mark.xfail(DASK_EXPR_ENABLED, reason='duplicated columns not supported')
def test_dataframe_mode():
    data = [['Tom', 10, 7], ['Farahn', 14, 7], ['Julie', 14, 5], ['Nick', 10, 10]]
    df = pd.DataFrame(data, columns=['Name', 'Num', 'Num'])
    ddf = dd.from_pandas(df, npartitions=3)
    assert_eq(ddf.mode(), df.mode())
    assert_eq(ddf.Name.mode(), df.Name.mode(), check_names=PANDAS_GE_140)
    df = pd.DataFrame(columns=['a', 'b'])
    ddf = dd.from_pandas(df, npartitions=1)
    assert_eq(ddf.mode(), df.mode(), check_index=False)