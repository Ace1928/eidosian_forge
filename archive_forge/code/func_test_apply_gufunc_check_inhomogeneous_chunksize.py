from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_check_inhomogeneous_chunksize():

    def foo(x, y):
        return x + y
    rng = da.random.default_rng()
    a = rng.normal(size=(8,), chunks=((2, 2, 2, 2),))
    b = rng.normal(size=(8,), chunks=((2, 3, 3),))
    with pytest.raises(ValueError) as excinfo:
        da.apply_gufunc(foo, '(),()->()', a, b, output_dtypes=float, allow_rechunk=False)
    assert 'with different chunksize present' in str(excinfo.value)