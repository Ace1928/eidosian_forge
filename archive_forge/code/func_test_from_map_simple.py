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
@pytest.mark.parametrize('vals', [('A', 'B'), (3, 4), (datetime(2020, 10, 1), datetime(2022, 12, 31))])
def test_from_map_simple(vals):

    def func(input, size=0):
        value, index = input
        return pd.Series([value] * size, index=[index] * size)
    iterable = [(vals[0], 1), (vals[1], 2)]
    ser = dd.from_map(func, iterable, size=2)
    expect = pd.Series([vals[0], vals[0], vals[1], vals[1]], index=[1, 1, 2, 2])
    if not DASK_EXPR_ENABLED:
        layers = ser.dask.layers
        expected_layers = 2 if pyarrow_strings_enabled() and any((isinstance(v, str) for v in vals)) else 1
        assert len(layers) == expected_layers
        assert isinstance(layers[ser._name], Blockwise)
    assert ser.npartitions == len(iterable)
    assert_eq(ser, expect)