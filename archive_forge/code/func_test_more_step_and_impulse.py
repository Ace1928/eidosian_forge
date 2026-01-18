import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
def test_more_step_and_impulse(self):
    lambda1 = 0.5
    lambda2 = 0.75
    a = np.array([[lambda1, 0.0], [0.0, lambda2]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    c = np.array([[1.0, 1.0]])
    d = np.array([[0.0, 0.0]])
    n = 10
    ts, ys = dstep((a, b, c, d, 1), n=n)
    stp0 = 1.0 / (1 - lambda1) * (1.0 - lambda1 ** np.arange(n))
    stp1 = 1.0 / (1 - lambda2) * (1.0 - lambda2 ** np.arange(n))
    assert_allclose(ys[0][:, 0], stp0)
    assert_allclose(ys[1][:, 0], stp1)
    x0 = np.array([1.0, 1.0])
    ti, yi = dimpulse((a, b, c, d, 1), n=n, x0=x0)
    imp = np.array([lambda1, lambda2]) ** np.arange(-1, n + 1).reshape(-1, 1)
    imp[0, :] = 0.0
    y0 = imp[:n, 0] + np.dot(imp[1:n + 1, :], x0)
    y1 = imp[:n, 1] + np.dot(imp[1:n + 1, :], x0)
    assert_allclose(yi[0][:, 0], y0)
    assert_allclose(yi[1][:, 0], y1)
    system = ([1.0], [1.0, -0.5], 0.1)
    t, (y,) = dstep(system, n=3)
    assert_allclose(t, [0, 0.1, 0.2])
    assert_array_equal(y.T, [[0, 1.0, 1.5]])
    t, (y,) = dimpulse(system, n=3)
    assert_allclose(t, [0, 0.1, 0.2])
    assert_array_equal(y.T, [[0, 1, 0.5]])