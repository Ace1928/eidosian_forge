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
@pytest.mark.gpu
@pytest.mark.parametrize('array_backend, df_backend', [('cupy', 'cudf'), ('numpy', 'pandas')])
def test_from_array_dispatching(array_backend, df_backend):
    array_lib = pytest.importorskip(array_backend)
    df_lib = pytest.importorskip(df_backend)
    with config.set({'array.backend': array_backend}):
        darr = da.ones(10)
    assert isinstance(darr._meta, array_lib.ndarray)
    ddf1 = dd.from_array(darr)
    ddf2 = dd.from_array(darr.compute())
    assert isinstance(ddf1._meta, df_lib.Series)
    assert isinstance(ddf2._meta, df_lib.Series)
    assert_eq(ddf1, ddf2)