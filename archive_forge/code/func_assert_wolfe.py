from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def assert_wolfe(s, phi, derphi, c1=0.0001, c2=0.9, err_msg=''):
    """
    Check that strong Wolfe conditions apply
    """
    phi1 = phi(s)
    phi0 = phi(0)
    derphi0 = derphi(0)
    derphi1 = derphi(s)
    msg = "s = {}; phi(0) = {}; phi(s) = {}; phi'(0) = {}; phi'(s) = {}; {}".format(s, phi0, phi1, derphi0, derphi1, err_msg)
    assert phi1 <= phi0 + c1 * s * derphi0, 'Wolfe 1 failed: ' + msg
    assert abs(derphi1) <= abs(c2 * derphi0), 'Wolfe 2 failed: ' + msg