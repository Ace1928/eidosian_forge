from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_first_order_fixed_time(self):
    system = ([1.0], [1.0, 1.0])
    n = 21
    t = np.linspace(0, 2.0, n)
    tout, y = self.func(system, T=t)
    assert_equal(tout.shape, (n,))
    assert_almost_equal(tout, t)
    expected_y = 1 - np.exp(-t)
    assert_almost_equal(y, expected_y)