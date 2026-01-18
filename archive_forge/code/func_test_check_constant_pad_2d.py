import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_check_constant_pad_2d(self):
    arr = np.arange(4).reshape(2, 2)
    test = np.lib.pad(arr, ((1, 2), (1, 3)), mode='constant', constant_values=((1, 2), (3, 4)))
    expected = np.array([[3, 1, 1, 4, 4, 4], [3, 0, 1, 4, 4, 4], [3, 2, 3, 4, 4, 4], [3, 2, 2, 4, 4, 4], [3, 2, 2, 4, 4, 4]])
    assert_allclose(test, expected)