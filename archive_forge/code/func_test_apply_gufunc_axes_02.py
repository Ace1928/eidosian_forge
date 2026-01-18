from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_axes_02():

    def matmul(x, y):
        return np.einsum('...ij,...jk->...ik', x, y)
    rng = np.random.default_rng()
    a = rng.standard_normal((3, 2, 1))
    b = rng.standard_normal((3, 7, 5))
    da_ = da.from_array(a, chunks=2)
    db = da.from_array(b, chunks=3)
    m = np.einsum('jiu,juk->uik', a, b)
    dm = apply_gufunc(matmul, '(i,j),(j,k)->(i,k)', da_, db, axes=[(1, 0), (0, -1), (-2, -1)], allow_rechunk=True)
    assert_eq(m, dm)