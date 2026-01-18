import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from scipy.stats import CensoredData
def test_left_censored(self):
    x = np.array([0, 3, 2.5])
    is_censored = np.array([0, 1, 0], dtype=bool)
    data = CensoredData.left_censored(x, is_censored)
    assert_equal(data._uncensored, x[~is_censored])
    assert_equal(data._left, x[is_censored])
    assert_equal(data._right, [])
    assert_equal(data._interval, np.empty((0, 2)))