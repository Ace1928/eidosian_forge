from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import dask
import dask.array as da
import dask.dataframe as dd
from dask import config
from dask.blockwise import Blockwise
from dask.dataframe._compat import PANDAS_GE_200, tm
from dask.dataframe.io.io import _meta_from_array
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import assert_eq, get_string_dtype, pyarrow_strings_enabled
from dask.delayed import Delayed, delayed
from dask.utils_test import hlg_layer_topological
def test_from_map_multi():
    func = lambda x, y: pd.DataFrame({'add': x + y})
    iterables = ([np.arange(2, dtype='int64'), np.arange(2, dtype='int64')], [np.array([2, 2], dtype='int64'), np.array([2, 2], dtype='int64')])
    index = np.array([0, 1, 0, 1], dtype='int64')
    expect = pd.DataFrame({'add': np.array([2, 3, 2, 3], dtype='int64')}, index=index)
    ddf = dd.from_map(func, *iterables)
    assert_eq(ddf, expect)