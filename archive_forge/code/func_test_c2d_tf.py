import numpy as np
from numpy.testing import \
import pytest
from scipy.signal import cont2discrete as c2d
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim, lti
from scipy.signal import tf2ss, impulse, dimpulse, step, dstep
def test_c2d_tf(self):
    sys = lti([0.5, 0.3], [1.0, 0.4])
    sys = sys.to_discrete(0.005)
    num_res = np.array([0.5, -0.485149004980066])
    den_res = np.array([1.0, -0.980198673306755])
    assert_allclose(sys.den, den_res, atol=0.02)
    assert_allclose(sys.num, num_res, atol=0.02)