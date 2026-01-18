from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_axis_02():

    def myfft(x):
        return np.fft.fft(x, axis=-1)
    a = np.random.default_rng().standard_normal((10, 5))
    da_ = da.from_array(a, chunks=2)
    m = np.fft.fft(a, axis=0)
    dm = apply_gufunc(myfft, '(i)->(i)', da_, axis=0, allow_rechunk=True)
    assert_eq(m, dm)