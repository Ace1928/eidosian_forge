from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_with_meta():

    def stats(x):
        return (np.mean(x, axis=-1), np.std(x, axis=-1, dtype=np.float32))
    a = da.random.default_rng().normal(size=(10, 20, 30), chunks=(5, 5, 30))
    meta = (np.ones(0, dtype=np.float64), np.ones(0, dtype=np.float32))
    result = apply_gufunc(stats, '(i)->(),()', a, meta=meta)
    expected = stats(a.compute())
    assert_eq(expected[0], result[0])
    assert_eq(expected[1], result[1])