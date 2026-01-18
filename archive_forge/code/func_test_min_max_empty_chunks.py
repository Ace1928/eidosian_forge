from __future__ import annotations
import os
import warnings
from contextlib import nullcontext as does_not_warn
from itertools import permutations, zip_longest
import pytest
import itertools
import dask.array as da
import dask.config as config
from dask.array.numpy_compat import NUMPY_GE_122, ComplexWarning
from dask.array.utils import assert_eq, same_keys
from dask.core import get_deps
@pytest.mark.parametrize(['dfunc', 'func'], [(da.min, np.min), (da.max, np.max)])
def test_min_max_empty_chunks(dfunc, func):
    x1 = np.arange(10)
    a1 = da.from_array(x1, chunks=1)
    assert_eq(dfunc(a1[a1 < 2]), func(x1[x1 < 2]))
    x2 = np.arange(10)
    a2 = da.from_array(x2, chunks=((5, 0, 5),))
    assert_eq(dfunc(a2), func(x2))
    x3 = np.array([[1, 1, 2, 3], [1, 1, 4, 0]])
    a3 = da.from_array(x3, chunks=1)
    assert_eq(dfunc(a3[a3 >= 2]), func(x3[x3 >= 2]))
    a4 = da.arange(10)
    with pytest.raises(ValueError):
        dfunc(a4[a4 < 0]).compute()