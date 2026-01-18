from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_via_numba_01():
    numba = pytest.importorskip('numba')

    @numba.guvectorize([(numba.float64[:], numba.float64[:], numba.float64[:])], '(n),(n)->(n)')
    def g(x, y, res):
        for i in range(x.shape[0]):
            res[i] = x[i] + y[i]
    rng = da.random.default_rng()
    a = rng.normal(size=(20, 30), chunks=30)
    b = rng.normal(size=(20, 30), chunks=30)
    x = a + b
    y = g(a, b, axis=0)
    assert_eq(x, y)