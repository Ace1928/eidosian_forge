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
def test_astype_categoricals_known():
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'b', 'c'], 'y': ['x', 'y', 'z', 'y', 'z'], 'z': ['b', 'b', 'b', 'c', 'b'], 'other': [1, 2, 3, 4, 5]})
    ddf = dd.from_pandas(df, 2)
    abc = pd.api.types.CategoricalDtype(['a', 'b', 'c'], ordered=False)
    category = pd.api.types.CategoricalDtype(ordered=False)
    ddf2 = ddf.astype({'x': abc, 'y': category, 'z': 'category', 'other': 'f8'})
    for col, known in [('x', True), ('y', False), ('z', False)]:
        x = getattr(ddf2, col)
        assert isinstance(x.dtype, pd.CategoricalDtype)
        assert x.cat.known == known
    for dtype, known in [('category', False), (category, False), (abc, True)]:
        dx2 = ddf.x.astype(dtype)
        assert isinstance(dx2.dtype, pd.CategoricalDtype)
        assert dx2.cat.known == known