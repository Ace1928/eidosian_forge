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
def test_from_array_raises_more_than_2D():
    x = da.ones((3, 3, 3), chunks=2)
    y = np.ones((3, 3, 3))
    with pytest.raises(ValueError, match='more than 2D array'):
        dd.from_dask_array(x)
    with pytest.raises(ValueError, match='more than 2D array'):
        dd.from_array(y)