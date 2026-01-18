from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_01b():

    def stats(x):
        return (np.mean(x, axis=-1), np.std(x, axis=-1))
    a = da.random.default_rng().normal(size=(10, 20, 30), chunks=5)
    mean, std = apply_gufunc(stats, '(i)->(),()', a, output_dtypes=2 * (a.dtype,), allow_rechunk=True)
    assert mean.compute().shape == (10, 20)
    assert std.compute().shape == (10, 20)