import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
def test_dlsim(self):
    a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
    b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
    c = np.asarray([[0.1, 0.3]])
    d = np.asarray([[0.0, -0.1, 0.0]])
    dt = 0.5
    u = np.hstack((np.linspace(0, 4.0, num=5)[:, np.newaxis], np.full((5, 1), 0.01), np.full((5, 1), -0.002)))
    t_in = np.linspace(0, 2.0, num=5)
    yout_truth = np.array([[-0.001, -0.00073, 0.039446, 0.0915387, 0.13195948]]).T
    xout_truth = np.asarray([[0, 0], [0.0012, 0.0005], [0.40233, 0.00071], [1.163368, -0.079327], [2.2402985, -0.3035679]])
    tout, yout, xout = dlsim((a, b, c, d, dt), u, t_in)
    assert_array_almost_equal(yout_truth, yout)
    assert_array_almost_equal(xout_truth, xout)
    assert_array_almost_equal(t_in, tout)
    dlsim((1, 2, 3), 4)
    u_sparse = u[[0, 4], :]
    t_sparse = np.asarray([0.0, 2.0])
    tout, yout, xout = dlsim((a, b, c, d, dt), u_sparse, t_sparse)
    assert_array_almost_equal(yout_truth, yout)
    assert_array_almost_equal(xout_truth, xout)
    assert_equal(len(tout), yout.shape[0])
    num = np.asarray([1.0, -0.1])
    den = np.asarray([0.3, 1.0, 0.2])
    yout_truth = np.array([[0.0, 0.0, 3.33333333333333, -4.77777777777778, 23.037037037037]]).T
    tout, yout = dlsim((num, den, 0.5), u[:, 0], t_in)
    assert_array_almost_equal(yout, yout_truth)
    assert_array_almost_equal(t_in, tout)
    uflat = np.asarray(u[:, 0])
    uflat = uflat.reshape((5,))
    tout, yout = dlsim((num, den, 0.5), uflat, t_in)
    assert_array_almost_equal(yout, yout_truth)
    assert_array_almost_equal(t_in, tout)
    zd = np.array([0.5, -0.5])
    pd = np.array([1j / np.sqrt(2), -1j / np.sqrt(2)])
    k = 1.0
    yout_truth = np.array([[0.0, 1.0, 2.0, 2.25, 2.5]]).T
    tout, yout = dlsim((zd, pd, k, 0.5), u[:, 0], t_in)
    assert_array_almost_equal(yout, yout_truth)
    assert_array_almost_equal(t_in, tout)
    system = lti([1], [1, 1])
    assert_raises(AttributeError, dlsim, system, u)