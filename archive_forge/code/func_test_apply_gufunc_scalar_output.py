from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_scalar_output():

    def foo():
        return 1
    x = apply_gufunc(foo, '->()', output_dtypes=int)
    assert x.compute() == 1