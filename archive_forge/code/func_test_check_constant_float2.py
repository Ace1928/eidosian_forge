import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_check_constant_float2(self):
    arr = np.arange(30).reshape(5, 6)
    arr_float = arr.astype(np.float64)
    test = np.pad(arr_float, ((1, 2), (1, 2)), mode='constant', constant_values=1.1)
    expected = np.array([[1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1], [1.1, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.1, 1.1], [1.1, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 1.1, 1.1], [1.1, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 1.1, 1.1], [1.1, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 1.1, 1.1], [1.1, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 1.1, 1.1], [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1], [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]])
    assert_allclose(test, expected)