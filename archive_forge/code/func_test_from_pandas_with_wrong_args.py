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
def test_from_pandas_with_wrong_args():
    df = pd.DataFrame({'x': [1, 2, 3]}, index=[3, 2, 1])
    with pytest.raises(TypeError, match='must be a pandas DataFrame or Series'):
        dd.from_pandas('foo')
    if not DASK_EXPR_ENABLED:
        with pytest.raises(ValueError, match='one of npartitions and chunksize must be specified'):
            dd.from_pandas(df)
    with pytest.raises(TypeError, match='provide npartitions as an int'):
        dd.from_pandas(df, npartitions=5.2, sort=False)
    with pytest.raises(TypeError, match='provide chunksize as an int'):
        dd.from_pandas(df, chunksize=18.27)