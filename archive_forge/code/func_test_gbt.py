import numpy as np
from numpy.testing import \
import pytest
from scipy.signal import cont2discrete as c2d
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim, lti
from scipy.signal import tf2ss, impulse, dimpulse, step, dstep
def test_gbt(self):
    ac = np.eye(2)
    bc = np.full((2, 1), 0.5)
    cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
    dc = np.array([[0.0], [0.0], [-0.33]])
    dt_requested = 0.5
    alpha = 1.0 / 3.0
    ad_truth = 1.6 * np.eye(2)
    bd_truth = np.full((2, 1), 0.3)
    cd_truth = np.array([[0.9, 1.2], [1.2, 1.2], [1.2, 0.3]])
    dd_truth = np.array([[0.175], [0.2], [-0.205]])
    ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested, method='gbt', alpha=alpha)
    assert_array_almost_equal(ad_truth, ad)
    assert_array_almost_equal(bd_truth, bd)
    assert_array_almost_equal(cd_truth, cd)
    assert_array_almost_equal(dd_truth, dd)