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
def test_merge_asof_with_empty():
    good_df = pd.DataFrame({'index_col': list(range(10)), 'good_val': list(range(10))})
    good_dd = dd.from_pandas(good_df, npartitions=2)
    empty_df = good_df[good_df.index_col < 0].copy().rename(columns={'good_val': 'empty_val'})
    empty_dd = dd.from_pandas(empty_df, npartitions=2)
    result_dd = dd.merge_asof(good_dd.set_index('index_col'), empty_dd.set_index('index_col'), left_index=True, right_index=True)
    result_df = pd.merge_asof(good_df.set_index('index_col'), empty_df.set_index('index_col'), left_index=True, right_index=True)
    assert_eq(result_dd, result_df, check_index=False)
    result_dd = dd.merge_asof(empty_dd.set_index('index_col'), good_dd.set_index('index_col'), left_index=True, right_index=True)
    result_df = pd.merge_asof(empty_df.set_index('index_col'), good_df.set_index('index_col'), left_index=True, right_index=True)
    assert_eq(result_dd, result_df, check_index=False)
    result_dd = dd.merge_asof(empty_dd.set_index('index_col'), empty_dd.set_index('index_col'), left_index=True, right_index=True)
    result_df = pd.merge_asof(empty_df.set_index('index_col'), empty_df.set_index('index_col'), left_index=True, right_index=True)
    assert_eq(result_dd, result_df, check_index=False)