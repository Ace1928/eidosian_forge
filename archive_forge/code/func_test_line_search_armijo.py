from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def test_line_search_armijo(self):
    c = 0
    for name, f, fprime, x, p, old_f in self.line_iter():
        f0 = f(x)
        g0 = fprime(x)
        self.fcount = 0
        s, fc, fv = ls.line_search_armijo(f, x, p, g0, f0)
        c += 1
        assert_equal(self.fcount, fc)
        assert_fp_equal(fv, f(x + s * p))
        assert_line_armijo(x, p, s, f, err_msg=name)
    assert c >= 9