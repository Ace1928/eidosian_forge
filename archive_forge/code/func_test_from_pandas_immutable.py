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
@pytest.mark.parametrize('index', [[1, 2, 3], [3, 2, 1]])
@pytest.mark.parametrize('sort', [True, False])
def test_from_pandas_immutable(sort, index):
    pdf = pd.DataFrame({'a': [1, 2, 3]}, index=index)
    expected = pdf.copy()
    df = dd.from_pandas(pdf, npartitions=2, sort=sort)
    pdf.iloc[0, 0] = 100
    assert_eq(df, expected)