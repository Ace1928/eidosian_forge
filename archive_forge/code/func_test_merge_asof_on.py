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
@pytest.mark.parametrize('allow_exact_matches', [True, False])
@pytest.mark.parametrize('direction', ['backward', 'forward', 'nearest'])
def test_merge_asof_on(allow_exact_matches, direction):
    A = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
    a = dd.from_pandas(A, npartitions=2)
    B = pd.DataFrame({'a': [1, 2, 3, 6, 7], 'right_val': [1, 2, 3, 6, 7]})
    b = dd.from_pandas(B, npartitions=2)
    C = pd.merge_asof(A, B, on='a', allow_exact_matches=allow_exact_matches, direction=direction)
    c = dd.merge_asof(a, b, on='a', allow_exact_matches=allow_exact_matches, direction=direction)
    assert_eq(c, C, check_index=False)