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
@pytest.mark.parametrize('how', ['right', 'outer'])
def test_merge_empty_left_df(shuffle_method, how):
    left = pd.DataFrame({'a': [1, 1, 2, 2], 'val': [5, 6, 7, 8]})
    right = pd.DataFrame({'a': [0, 0, 3, 3], 'val': [11, 12, 13, 14]})
    dd_left = dd.from_pandas(left, npartitions=4)
    dd_right = dd.from_pandas(right, npartitions=4)
    merged = dd_left.merge(dd_right, on='a', how=how)
    expected = left.merge(right, on='a', how=how)
    assert_eq(merged, expected, check_index=False)
    merged.map_partitions(lambda x: x, meta=merged._meta).compute()