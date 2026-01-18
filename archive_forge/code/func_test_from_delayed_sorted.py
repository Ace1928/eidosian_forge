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
@pytest.mark.xfail(DASK_EXPR_ENABLED, reason='not supported')
def test_from_delayed_sorted():
    a = pd.DataFrame({'x': [1, 2]}, index=[1, 10])
    b = pd.DataFrame({'x': [4, 1]}, index=[100, 200])
    A = dd.from_delayed([delayed(a), delayed(b)], divisions='sorted')
    assert A.known_divisions
    assert A.divisions == (1, 100, 200)