import numpy as np
from numpy.testing import \
import pytest
from scipy.signal import cont2discrete as c2d
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim, lti
from scipy.signal import tf2ss, impulse, dimpulse, step, dstep
def test_c2d_ss(self):
    A = np.array([[-0.3, 0.1], [0.2, -0.7]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = 0
    A_res = np.array([[0.985136404135682, 0.004876671474795], [0.00975334294959, 0.965629718236502]])
    B_res = np.array([[0.000122937599964], [0.049135527547844]])
    sys_ssc = lti(A, B, C, D)
    sys_ssd = sys_ssc.to_discrete(0.05)
    assert_allclose(sys_ssd.A, A_res)
    assert_allclose(sys_ssd.B, B_res)
    assert_allclose(sys_ssd.C, C)
    assert_allclose(sys_ssd.D, D)