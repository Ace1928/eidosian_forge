import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_in1d_hit_alternate_algorithm(self):
    """Hit the standard isin code with integers"""
    a = np.array([5, 4, 5, 3, 4, 4, 1000000000.0], dtype=np.int64)
    b = np.array([2, 3, 4, 1000000000.0], dtype=np.int64)
    expected = np.array([0, 1, 0, 1, 1, 1, 1], dtype=bool)
    assert_array_equal(expected, in1d(a, b))
    assert_array_equal(np.invert(expected), in1d(a, b, invert=True))
    a = np.array([5, 7, 1, 2], dtype=np.int64)
    b = np.array([2, 4, 3, 1, 5, 1000000000.0], dtype=np.int64)
    ec = np.array([True, False, True, True])
    c = in1d(a, b, assume_unique=True)
    assert_array_equal(c, ec)