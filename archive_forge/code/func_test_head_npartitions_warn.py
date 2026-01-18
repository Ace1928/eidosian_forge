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
def test_head_npartitions_warn():
    match = '5 elements requested, only 3 elements'
    with pytest.warns(UserWarning, match=match):
        d.head(5)
    match = 'Insufficient elements'
    with pytest.warns(UserWarning, match=match):
        d.head(100)
    with pytest.warns(UserWarning, match=match):
        d.head(7)
    with pytest.warns(UserWarning, match=match):
        d.head(7, npartitions=2)
    for n in [3, -1]:
        with warnings.catch_warnings(record=True) as record:
            d.head(10, npartitions=n)
        assert not record
    d2 = dd.from_pandas(pd.DataFrame({'x': [1, 2, 3]}), npartitions=1)
    with warnings.catch_warnings(record=True) as record:
        d2.head()
    assert not record