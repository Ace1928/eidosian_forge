from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def test_armijo_terminate_1(self):
    count = [0]

    def phi(s):
        count[0] += 1
        return -s + 0.01 * s ** 2
    s, phi1 = ls.scalar_search_armijo(phi, phi(0), -1, alpha0=1)
    assert_equal(s, 1)
    assert_equal(count[0], 2)
    assert_armijo(s, phi)