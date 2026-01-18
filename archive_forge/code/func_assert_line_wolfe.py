from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def assert_line_wolfe(x, p, s, f, fprime, **kw):
    assert_wolfe(s, phi=lambda sp: f(x + p * sp), derphi=lambda sp: np.dot(fprime(x + p * sp), p), **kw)