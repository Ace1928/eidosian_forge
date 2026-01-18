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
def test_cheap_single_partition_merge_divisions():
    a = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6], 'y': list('abdabd')}, index=[10, 20, 30, 40, 50, 60])
    aa = dd.from_pandas(a, npartitions=3)
    b = pd.DataFrame({'x': [1, 2, 3, 4], 'z': list('abda')})
    bb = dd.from_pandas(b, npartitions=1, sort=False)
    actual = aa.merge(bb, on='x', how='inner')
    if not DASK_EXPR_ENABLED:
        assert not hlg_layer_topological(actual.dask, -1).is_materialized()
    assert not actual.known_divisions
    assert_divisions(actual)
    actual = bb.merge(aa, on='x', how='inner')
    assert not actual.known_divisions
    assert_divisions(actual)