import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_2d_without_axis(self):
    i, j = np.ogrid[:4, :4]
    arr = 10 * i + j
    shape = (2, 3)
    arr_view = sliding_window_view(arr, shape)
    expected = np.array([[[[0, 1, 2], [10, 11, 12]], [[1, 2, 3], [11, 12, 13]]], [[[10, 11, 12], [20, 21, 22]], [[11, 12, 13], [21, 22, 23]]], [[[20, 21, 22], [30, 31, 32]], [[21, 22, 23], [31, 32, 33]]]])
    assert_array_equal(arr_view, expected)