from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_gufunc_mixed_inputs_vectorize():

    def foo(x, y):
        return (x + y).sum(axis=1)
    a = da.ones((8, 3, 5), chunks=(2, 3, 5), dtype=int)
    b = np.ones(5, dtype=int)
    x = apply_gufunc(foo, '(m,n),(n)->(m)', a, b, vectorize=True)
    assert_eq(x, np.full((8, 3), 10, dtype=int))