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
@pytest.mark.parametrize('unknown_divisions', [False, True])
def test_merge_asof_left_on_right_index(allow_exact_matches, direction, unknown_divisions):
    A = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']}, index=[10, 20, 30])
    a = dd.from_pandas(A, npartitions=2)
    B = pd.DataFrame({'right_val': [2, 3, 6, 7]}, index=[2, 3, 6, 7])
    b = dd.from_pandas(B, npartitions=2)
    if unknown_divisions:
        a = a.clear_divisions()
    C = pd.merge_asof(A, B, left_on='a', right_index=True, allow_exact_matches=allow_exact_matches, direction=direction)
    c = dd.merge_asof(a, b, left_on='a', right_index=True, allow_exact_matches=allow_exact_matches, direction=direction)
    assert_eq(c, C)
    for nparts in [1, 2, 3]:
        for a1, idx2 in (([5, 10, 15, 20], [1, 2, 3, 4]), ([1, 2, 3, 4], [5, 10, 15, 20]), ([5, 5, 10, 10, 15, 15], [4, 5, 6, 9, 10, 11, 14, 15, 16]), ([5, 10, 15], [4, 4, 5, 5, 6, 6, 9, 9, 10, 10, 11, 11])):
            A = pd.DataFrame({'a': a1}, index=[x * 10 for x in a1])
            a = dd.from_pandas(A, npartitions=nparts)
            B = pd.DataFrame({'b': idx2}, index=idx2)
            b = dd.from_pandas(B, npartitions=nparts)
            if unknown_divisions:
                a = a.clear_divisions()
            C = pd.merge_asof(A, B, left_on='a', right_index=True, allow_exact_matches=allow_exact_matches, direction=direction)
            c = dd.merge_asof(a, b, left_on='a', right_index=True, allow_exact_matches=allow_exact_matches, direction=direction)
            assert_eq(c, C)