import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_2d_with_axis(self):
    i, j = np.ogrid[:3, :4]
    arr = 10 * i + j
    arr_view = sliding_window_view(arr, 3, 0)
    expected = np.array([[[0, 10, 20], [1, 11, 21], [2, 12, 22], [3, 13, 23]]])
    assert_array_equal(arr_view, expected)