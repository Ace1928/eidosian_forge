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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not available in dask-expr')
@pytest.mark.parametrize('shuffle', [None, 'tasks'])
def test_broadcast_true(shuffle):
    left = dd.from_dict({'a': [1, 2] * 80, 'b_left': range(160)}, npartitions=16)
    right = dd.from_dict({'a': [2, 1] * 10, 'b_right': range(20)}, npartitions=2)
    result = dd.merge(left, right, broadcast=True, shuffle_method=shuffle)
    assert hlg_layer(result.dask, 'bcast-join')