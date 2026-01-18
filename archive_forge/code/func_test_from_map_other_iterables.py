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
@pytest.mark.parametrize('iterable', [enumerate(['A', 'B', 'C']), ((0, 'A'), (1, 'B'), (2, 'C')), _generator()])
def test_from_map_other_iterables(iterable):

    def func(t):
        size = t[0] + 1
        x = t[1]
        return pd.Series([x] * size)
    ddf = dd.from_map(func, iterable)
    expect = pd.Series(['A', 'B', 'B', 'C', 'C', 'C'], index=[0, 0, 1, 0, 1, 2])
    assert_eq(ddf.compute(), expect)