from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_elemwise_02():

    def addmul(x, y):
        assert x.shape in ((2,), (1,))
        return (x + y, x * y)
    a = da.from_array(np.array([1, 2, 3]), chunks=2, name='a')
    b = da.from_array(np.array([1, 2, 3]), chunks=2, name='b')
    z1, z2 = apply_gufunc(addmul, '(),()->(),()', a, b, output_dtypes=2 * (a.dtype,))
    assert_eq(z1, np.array([2, 4, 6]))
    assert_eq(z2, np.array([1, 4, 9]))