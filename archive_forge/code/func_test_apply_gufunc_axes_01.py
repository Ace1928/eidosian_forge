from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
@pytest.mark.parametrize('axes', [[0, 1], [(0,), (1,)]])
def test_apply_gufunc_axes_01(axes):

    def mystats(x, y):
        return np.std(x, axis=-1) * np.mean(y, axis=-1)
    rng = np.random.default_rng()
    a = rng.standard_normal((10, 5))
    b = rng.standard_normal((5, 6))
    da_ = da.from_array(a, chunks=2)
    db_ = da.from_array(b, chunks=2)
    m = np.std(a, axis=0) * np.mean(b, axis=1)
    dm = apply_gufunc(mystats, '(i),(j)->()', da_, db_, axes=axes, allow_rechunk=True)
    assert_eq(m, dm)