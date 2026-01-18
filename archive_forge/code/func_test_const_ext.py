import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises as assert_raises
from scipy.signal._arraytools import (axis_slice, axis_reverse,
def test_const_ext(self):
    a = np.array([[1, 2, 3, 4, 5], [9, 8, 7, 6, 5]])
    const = const_ext(a, 2, axis=1)
    expected = np.array([[1, 1, 1, 2, 3, 4, 5, 5, 5], [9, 9, 9, 8, 7, 6, 5, 5, 5]])
    assert_array_equal(const, expected)
    const = const_ext(a, 1, axis=0)
    expected = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [9, 8, 7, 6, 5], [9, 8, 7, 6, 5]])
    assert_array_equal(const, expected)