import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_in1d_with_arrays_containing_tuples(self):
    ar1 = np.array([(1,), 2], dtype=object)
    ar2 = np.array([(1,), 2], dtype=object)
    expected = np.array([True, True])
    result = np.in1d(ar1, ar2)
    assert_array_equal(result, expected)
    result = np.in1d(ar1, ar2, invert=True)
    assert_array_equal(result, np.invert(expected))
    ar1 = np.array([(1,), (2, 1), 1], dtype=object)
    ar1 = ar1[:-1]
    ar2 = np.array([(1,), (2, 1), 1], dtype=object)
    ar2 = ar2[:-1]
    expected = np.array([True, True])
    result = np.in1d(ar1, ar2)
    assert_array_equal(result, expected)
    result = np.in1d(ar1, ar2, invert=True)
    assert_array_equal(result, np.invert(expected))
    ar1 = np.array([(1,), (2, 3), 1], dtype=object)
    ar1 = ar1[:-1]
    ar2 = np.array([(1,), 2], dtype=object)
    expected = np.array([True, False])
    result = np.in1d(ar1, ar2)
    assert_array_equal(result, expected)
    result = np.in1d(ar1, ar2, invert=True)
    assert_array_equal(result, np.invert(expected))