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
@pytest.mark.parametrize('func', [np.cumsum, np.cumprod])
def test_array_cumreduction_out(func):
    x = da.ones((10, 10), chunks=(4, 4))
    func(x, axis=0, out=x)
    assert_eq(x, func(np.ones((10, 10)), axis=0))