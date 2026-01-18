import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from scipy.stats import CensoredData
def test_interval_censored_basic(self):
    a = [0.5, 2.0, 3.0, 5.5]
    b = [1.0, 2.5, 3.5, 7.0]
    data = CensoredData.interval_censored(low=a, high=b)
    assert_array_equal(data._interval, np.array(list(zip(a, b))))
    assert data._uncensored.shape == (0,)
    assert data._left.shape == (0,)
    assert data._right.shape == (0,)