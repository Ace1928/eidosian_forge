import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
def test_from_state_space(self):
    system_TF = dlti([2], [1, -0.5, 0, 0])
    A = np.array([[0.5, 0, 0], [1, 0, 0], [0, 1, 0]])
    B = np.array([[1, 0, 0]]).T
    C = np.array([[0, 0, 2]])
    D = 0
    system_SS = dlti(A, B, C, D)
    w = 10.0 ** np.arange(-3, 0, 0.5)
    with suppress_warnings() as sup:
        sup.filter(BadCoefficients)
        w1, H1 = dfreqresp(system_TF, w=w)
        w2, H2 = dfreqresp(system_SS, w=w)
    assert_almost_equal(H1, H2)