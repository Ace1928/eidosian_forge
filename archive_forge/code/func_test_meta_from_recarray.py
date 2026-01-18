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
def test_meta_from_recarray():
    x = np.array([(i, i * 10) for i in range(10)], dtype=[('a', np.float64), ('b', np.int64)])
    res = _meta_from_array(x)
    assert isinstance(res, pd.DataFrame)
    assert res['a'].dtype == np.float64
    assert res['b'].dtype == np.int64
    tm.assert_index_equal(res.columns, pd.Index(['a', 'b']))
    res = _meta_from_array(x, columns=['b', 'a'])
    assert isinstance(res, pd.DataFrame)
    assert res['a'].dtype == np.float64
    assert res['b'].dtype == np.int64
    tm.assert_index_equal(res.columns, pd.Index(['b', 'a']))
    with pytest.raises(ValueError):
        _meta_from_array(x, columns=['a', 'b', 'c'])