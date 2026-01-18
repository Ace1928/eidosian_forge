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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not available')
@pytest.mark.parametrize('how', ['inner', 'left', 'right', 'outer'])
def test_hash_join(how, shuffle_method):
    A = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6], 'y': [1, 1, 2, 2, 3, 4]})
    a = dd.repartition(A, [0, 4, 5])
    B = pd.DataFrame({'y': [1, 3, 4, 4, 5, 6], 'z': [6, 5, 4, 3, 2, 1]})
    b = dd.repartition(B, [0, 2, 5])
    c = hash_join(a, 'y', b, 'y', how)
    assert not hlg_layer_topological(c.dask, -1).is_materialized()
    result = c.compute()
    expected = pd.merge(A, B, how, 'y')
    list_eq(result, expected)
    c = hash_join(a, 'x', b, 'z', 'outer', npartitions=3, shuffle_method=shuffle_method)
    assert not hlg_layer_topological(c.dask, -1).is_materialized()
    assert c.npartitions == 3
    result = c.compute(scheduler='single-threaded')
    expected = pd.merge(A, B, 'outer', None, 'x', 'z')
    list_eq(result, expected)
    assert hash_join(a, 'y', b, 'y', 'inner', shuffle_method=shuffle_method)._name == hash_join(a, 'y', b, 'y', 'inner', shuffle_method=shuffle_method)._name
    assert hash_join(a, 'y', b, 'y', 'inner', shuffle_method=shuffle_method)._name != hash_join(a, 'y', b, 'y', 'outer', shuffle_method=shuffle_method)._name