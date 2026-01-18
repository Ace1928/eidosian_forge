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
def test_from_pandas_with_datetime_index():
    df = pd.DataFrame({'Date': ['2015-08-28', '2015-08-27', '2015-08-26', '2015-08-25', '2015-08-24', '2015-08-21', '2015-08-20', '2015-08-19', '2015-08-18'], 'Val': list(range(9))})
    df.Date = df.Date.astype('datetime64[ns]')
    ddf = dd.from_pandas(df, 2)
    assert_eq(df, ddf)
    ddf = dd.from_pandas(df, chunksize=2)
    assert_eq(df, ddf)