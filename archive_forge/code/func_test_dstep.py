import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
def test_dstep(self):
    a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
    b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
    c = np.asarray([[0.1, 0.3]])
    d = np.asarray([[0.0, -0.1, 0.0]])
    dt = 0.5
    yout_step_truth = (np.asarray([0.0, 0.04, 0.052, 0.0404, 0.00956, -0.036324, -0.093318, -0.15782348, -0.226628324, -0.2969374948]), np.asarray([-0.1, -0.075, -0.058, -0.04815, -0.04453, -0.0461895, -0.0521812, -0.061588875, -0.073549579, -0.08727047595]), np.asarray([0.0, -0.01, -0.013, -0.0101, -0.00239, 0.009081, 0.0233295, 0.03945587, 0.056657081, 0.0742343737]))
    tout, yout = dstep((a, b, c, d, dt), n=10)
    assert_equal(len(yout), 3)
    for i in range(0, len(yout)):
        assert_equal(yout[i].shape[0], 10)
        assert_array_almost_equal(yout[i].flatten(), yout_step_truth[i])
    tfin = ([1.0], [1.0, 1.0], 0.5)
    yout_tfstep = np.asarray([0.0, 1.0, 0.0])
    tout, yout = dstep(tfin, n=3)
    assert_equal(len(yout), 1)
    assert_array_almost_equal(yout[0].flatten(), yout_tfstep)
    zpkin = tf2zpk(tfin[0], tfin[1]) + (0.5,)
    tout, yout = dstep(zpkin, n=3)
    assert_equal(len(yout), 1)
    assert_array_almost_equal(yout[0].flatten(), yout_tfstep)
    system = lti([1], [1, 1])
    assert_raises(AttributeError, dstep, system)