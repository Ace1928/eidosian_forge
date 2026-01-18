from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_double_integrator(self):
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[2.0, 0.0]])
    system = self.lti_nowarn(A, B, C, 0.0)
    t = np.linspace(0, 5)
    u = np.ones_like(t)
    tout, y, x = self.func(system, u, t)
    expected_x = np.transpose(np.array([0.5 * tout ** 2, tout]))
    expected_y = tout ** 2
    assert_almost_equal(x, expected_x, decimal=self.digits_accuracy)
    assert_almost_equal(y, expected_y, decimal=self.digits_accuracy)