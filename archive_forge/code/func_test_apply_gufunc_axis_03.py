from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_axis_03():

    def mydiff(x):
        return np.diff(x, axis=-1)
    a = np.random.default_rng().standard_normal((3, 6, 4))
    da_ = da.from_array(a, chunks=2)
    m = np.diff(a, axis=1)
    dm = apply_gufunc(mydiff, '(i)->(i)', da_, axis=1, output_sizes={'i': 5}, allow_rechunk=True)
    assert_eq(m, dm)