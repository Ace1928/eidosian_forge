import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_check_2d(self):
    arr = np.arange(20).reshape(4, 5).astype(np.float64)
    test = np.pad(arr, (2, 2), mode='linear_ramp', end_values=(0, 0))
    expected = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 0.0], [0.0, 2.5, 5.0, 6.0, 7.0, 8.0, 9.0, 4.5, 0.0], [0.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 7.0, 0.0], [0.0, 7.5, 15.0, 16.0, 17.0, 18.0, 19.0, 9.5, 0.0], [0.0, 3.75, 7.5, 8.0, 8.5, 9.0, 9.5, 4.75, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert_allclose(test, expected)