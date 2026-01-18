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
@pytest.mark.parametrize('func', ['nanvar', 'nanstd'])
def test_nan_func_does_not_warn(func):
    x = np.ones((10,)) * np.nan
    x[0] = 1
    x[1] = 2
    d = da.from_array(x, chunks=2)
    with warnings.catch_warnings(record=True) as rec:
        getattr(da, func)(d).compute()
    assert not rec