from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_pass_additional_kwargs():

    def foo(x, bar):
        assert bar == 2
        return x
    ret = apply_gufunc(foo, '()->()', 1.0, output_dtypes=float, bar=2)
    assert_eq(ret, np.array(1.0, dtype=float))