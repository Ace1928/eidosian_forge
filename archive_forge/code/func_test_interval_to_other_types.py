import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from scipy.stats import CensoredData
def test_interval_to_other_types(self):
    interval = np.array([[0, 1], [2, 2], [3, 3], [9, np.inf], [8, np.inf], [-np.inf, 0], [1, 2]])
    data = CensoredData(interval=interval)
    assert_equal(data._uncensored, [2, 3])
    assert_equal(data._left, [0])
    assert_equal(data._right, [9, 8])
    assert_equal(data._interval, [[0, 1], [1, 2]])