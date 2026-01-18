from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_02():

    def outer_product(x, y):
        return np.einsum('...i,...j->...ij', x, y)
    rng = da.random.default_rng()
    a = rng.normal(size=(20, 30), chunks=(5, 30))
    b = rng.normal(size=(10, 1, 40), chunks=(10, 1, 40))
    c = apply_gufunc(outer_product, '(i),(j)->(i,j)', a, b, output_dtypes=a.dtype)
    assert c.compute().shape == (10, 20, 30, 40)