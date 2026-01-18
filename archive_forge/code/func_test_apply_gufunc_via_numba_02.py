from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_via_numba_02():
    numba = pytest.importorskip('numba')

    @numba.guvectorize([(numba.float64[:], numba.float64[:])], '(n)->()')
    def mysum(x, res):
        res[0] = 0.0
        for i in range(x.shape[0]):
            res[0] += x[i]
    a = da.random.default_rng().normal(size=(20, 30), chunks=30)
    x = a.sum(axis=0, keepdims=True)
    y = mysum(a, axis=0, keepdims=True)
    assert_eq(x, y)