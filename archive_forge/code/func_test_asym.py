from numpy.testing import (
from numpy import (
import numpy as np
import pytest
def test_asym(self):
    x = array([1, 1, 2, 3, 4, 4, 4, 5])
    y = array([1, 3, 2, 0, 1, 2, 3, 4])
    H, xed, yed = histogram2d(x, y, (6, 5), range=[[0, 6], [0, 5]], density=True)
    answer = array([[0.0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 1]])
    assert_array_almost_equal(H, answer / 8.0, 3)
    assert_array_equal(xed, np.linspace(0, 6, 7))
    assert_array_equal(yed, np.linspace(0, 5, 6))