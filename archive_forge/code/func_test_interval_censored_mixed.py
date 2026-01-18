import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from scipy.stats import CensoredData
def test_interval_censored_mixed(self):
    a = [0.5, -np.inf, -13.0, 2.0, 1.0, 10.0, -1.0]
    b = [0.5, 2500.0, np.inf, 3.0, 1.0, 11.0, np.inf]
    data = CensoredData.interval_censored(low=a, high=b)
    assert_array_equal(data._interval, [[2.0, 3.0], [10.0, 11.0]])
    assert_array_equal(data._uncensored, [0.5, 1.0])
    assert_array_equal(data._left, [2500.0])
    assert_array_equal(data._right, [-13.0, -1.0])