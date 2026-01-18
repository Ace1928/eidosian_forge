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
def test_merge_index_without_divisions(shuffle_method):
    a = pd.DataFrame({'x': [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
    b = pd.DataFrame({'y': [1, 2, 3, 4, 5]}, index=[5, 4, 3, 2, 1])
    aa = dd.from_pandas(a, npartitions=3, sort=False)
    bb = dd.from_pandas(b, npartitions=2)
    result = aa.join(bb, how='inner', shuffle_method=shuffle_method)
    expected = a.join(b, how='inner')
    assert_eq(result, expected)