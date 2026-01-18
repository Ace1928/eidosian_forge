import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def assert_same_as_ufunc(shape0, shape1, transposed=False, flipped=False):
    x0 = np.zeros(shape0, dtype=int)
    n = int(np.multiply.reduce(shape1))
    x1 = np.arange(n).reshape(shape1)
    if transposed:
        x0 = x0.T
        x1 = x1.T
    if flipped:
        x0 = x0[::-1]
        x1 = x1[::-1]
    y = x0 + x1
    b0, b1 = broadcast_arrays(x0, x1)
    assert_array_equal(y, b1)