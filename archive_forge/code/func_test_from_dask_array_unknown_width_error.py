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
def test_from_dask_array_unknown_width_error():
    dsk = {('x', 0, 0): np.random.random((2, 3)), ('x', 1, 0): np.random.random((5, 3))}
    dx = da.Array(dsk, 'x', ((np.nan, np.nan), (np.nan,)), np.float64)
    with pytest.raises(ValueError, match='Shape along axis 1 must be known'):
        dd.from_dask_array(dx)