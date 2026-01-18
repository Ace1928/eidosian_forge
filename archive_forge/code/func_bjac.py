import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy.integrate import odeint
import scipy.integrate._test_odeint_banded as banded5x5
def bjac(y, t):
    n = len(y)
    bjac = np.zeros((4, n), order='F')
    banded5x5.banded5x5_bjac(t, y, 1, 1, bjac)
    return bjac