import numpy as np
from numpy.testing import \
import pytest
from scipy.signal import cont2discrete as c2d
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim, lti
from scipy.signal import tf2ss, impulse, dimpulse, step, dstep
def test_discrete_approx(self):
    """
        Test that the solution to the discrete approximation of a continuous
        system actually approximates the solution to the continuous system.
        This is an indirect test of the correctness of the implementation
        of cont2discrete.
        """

    def u(t):
        return np.sin(2.5 * t)
    a = np.array([[-0.01]])
    b = np.array([[1.0]])
    c = np.array([[1.0]])
    d = np.array([[0.2]])
    x0 = 1.0
    t = np.linspace(0, 10.0, 101)
    dt = t[1] - t[0]
    u1 = u(t)
    t, yout, xout = lsim((a, b, c, d), T=t, U=u1, X0=x0)
    dsys = c2d((a, b, c, d), dt, method='bilinear')
    u2 = 0.5 * (u1[:-1] + u1[1:])
    t2 = t[:-1]
    td2, yd2, xd2 = dlsim(dsys, u=u2.reshape(-1, 1), t=t2, x0=x0)
    ymid = 0.5 * (yout[:-1] + yout[1:])
    assert_allclose(yd2.ravel(), ymid, rtol=0.0001)