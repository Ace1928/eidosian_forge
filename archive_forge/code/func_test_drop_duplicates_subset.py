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
def test_drop_duplicates_subset():
    df = pd.DataFrame({'x': [1, 2, 3, 1, 2, 3], 'y': ['a', 'a', 'b', 'b', 'c', 'c']})
    ddf = dd.from_pandas(df, npartitions=2)
    for kwarg in [{'keep': 'first'}, {'keep': 'last'}]:
        assert_eq(df.x.drop_duplicates(**kwarg), ddf.x.drop_duplicates(**kwarg))
        for ss in [['x'], 'y', ['x', 'y']]:
            assert_eq(df.drop_duplicates(subset=ss, **kwarg), ddf.drop_duplicates(subset=ss, **kwarg))
            assert_eq(df.drop_duplicates(ss, **kwarg), ddf.drop_duplicates(ss, **kwarg))