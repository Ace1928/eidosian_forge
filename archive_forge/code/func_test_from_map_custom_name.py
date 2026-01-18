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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="dask-expr doesn't support token")
def test_from_map_custom_name():
    func = lambda x: pd.DataFrame({'x': [x] * 2})
    iterable = ['A', 'B']
    label = 'my-label'
    token = '8675309'
    expect = pd.DataFrame({'x': ['A', 'A', 'B', 'B']}, index=[0, 1, 0, 1])
    ddf = dd.from_map(func, iterable, label=label, token=token)
    if not pyarrow_strings_enabled():
        assert ddf._name == label + '-' + token
    assert_eq(ddf, expect)