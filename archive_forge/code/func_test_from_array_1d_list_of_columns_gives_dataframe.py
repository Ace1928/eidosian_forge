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
def test_from_array_1d_list_of_columns_gives_dataframe():
    x = da.ones(10, chunks=3)
    y = np.ones(10)
    d1 = dd.from_dask_array(x, columns=['name'])
    p1 = pd.DataFrame(y, columns=['name'])
    assert_eq(d1, p1)
    d2 = dd.from_array(y, columns=['name'])
    assert_eq(d2, d1)