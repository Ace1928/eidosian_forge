import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.special import logit, expit, log_expit
def test_large_negative(self):
    x = np.array([-10000.0, -750.0, -500.0, -35.0])
    y = log_expit(x)
    assert_equal(y, x)