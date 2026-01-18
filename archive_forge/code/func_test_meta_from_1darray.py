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
def test_meta_from_1darray():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    res = _meta_from_array(x)
    assert isinstance(res, pd.Series)
    assert res.dtype == np.float64
    x = np.array([1, 2, 3], dtype=np.object_)
    res = _meta_from_array(x, columns='x')
    assert isinstance(res, pd.Series)
    assert res.name == 'x'
    assert res.dtype == np.object_
    x = np.array([1, 2, 3], dtype=np.object_)
    res = _meta_from_array(x, columns=['x'])
    assert isinstance(res, pd.DataFrame)
    assert res['x'].dtype == np.object_
    tm.assert_index_equal(res.columns, pd.Index(['x']))
    with pytest.raises(ValueError):
        _meta_from_array(x, columns=['a', 'b'])