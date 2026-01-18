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
def test_from_dask_array_struct_dtype():
    x = np.array([(1, 'a'), (2, 'b')], dtype=[('a', 'i4'), ('b', 'object')])
    y = da.from_array(x, chunks=(1,))
    df = dd.from_dask_array(y)
    tm.assert_index_equal(df.columns, pd.Index(['a', 'b']))
    assert_eq(df, pd.DataFrame(x))
    assert_eq(dd.from_dask_array(y, columns=['b', 'a']), pd.DataFrame(x, columns=['b', 'a']))