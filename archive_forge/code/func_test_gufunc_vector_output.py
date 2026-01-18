from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_gufunc_vector_output():

    def foo():
        return np.array([1, 2, 3], dtype=int)
    x = apply_gufunc(foo, '->(i_0)', output_dtypes=int, output_sizes={'i_0': 3})
    assert x.chunks == ((3,),)
    assert_eq(x, np.array([1, 2, 3]))