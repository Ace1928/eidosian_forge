from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.api.types import is_object_dtype
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.core import _Frame
from dask.dataframe.methods import concat
from dask.dataframe.multi import (
from dask.dataframe.utils import (
from dask.utils_test import hlg_layer, hlg_layer_topological
@pytest.mark.parametrize('left_col, right_col', [('endofweek', 'timestamp'), ('endofweek', 'endofweek')])
def test_merge_asof_on_left_right(left_col, right_col):
    df1 = pd.DataFrame({left_col: [1, 1, 2, 2, 3, 4], 'GroupCol': [1234, 8679, 1234, 8679, 1234, 8679]})
    df2 = pd.DataFrame({right_col: [0, 0, 1, 3], 'GroupVal': [1234, 1234, 8679, 1234]})
    result_df = pd.merge_asof(df1, df2, left_on=left_col, right_on=right_col)
    result_dd = dd.merge_asof(dd.from_pandas(df1, npartitions=2), dd.from_pandas(df2, npartitions=2), left_on=left_col, right_on=right_col)
    assert_eq(result_df, result_dd, check_index=False)