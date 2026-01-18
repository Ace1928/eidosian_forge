from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_gufunc_two_inputs():

    def foo(x, y):
        return np.einsum('...ij,...jk->ik', x, y)
    a = da.ones((2, 3), chunks=100, dtype=int)
    b = da.ones((3, 4), chunks=100, dtype=int)
    x = apply_gufunc(foo, '(i,j),(j,k)->(i,k)', a, b, output_dtypes=int)
    assert_eq(x, 3 * np.ones((2, 4), dtype=int))