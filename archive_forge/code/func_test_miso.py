from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_miso(self):
    A = np.array([[-1.0, 0.0], [0.0, -2.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    C = np.array([1.0, 0.0])
    D = np.zeros((1, 2))
    system = self.lti_nowarn(A, B, C, D)
    t = np.linspace(0, 5.0, 101)
    u = np.zeros((len(t), 2))
    tout, y, x = self.func(system, u, t, X0=[1.0, 1.0])
    expected_y = np.exp(-tout)
    expected_x0 = np.exp(-tout)
    expected_x1 = np.exp(-2.0 * tout)
    assert_almost_equal(y, expected_y)
    assert_almost_equal(x[:, 0], expected_x0)
    assert_almost_equal(x[:, 1], expected_x1)