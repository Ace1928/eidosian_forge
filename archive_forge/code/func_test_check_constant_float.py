import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_check_constant_float(self):
    arr = np.arange(30).reshape(5, 6)
    test = np.pad(arr, (1, 2), mode='constant', constant_values=1.1)
    expected = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 2, 3, 4, 5, 1, 1], [1, 6, 7, 8, 9, 10, 11, 1, 1], [1, 12, 13, 14, 15, 16, 17, 1, 1], [1, 18, 19, 20, 21, 22, 23, 1, 1], [1, 24, 25, 26, 27, 28, 29, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
    assert_allclose(test, expected)