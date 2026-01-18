from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def test_line_search_wolfe2_bounds(self):

    def f(x):
        return np.dot(x, x)

    def fp(x):
        return 2 * x
    p = np.array([1, 0])
    x = -60 * p
    c2 = 0.5
    s, _, _, _, _, _ = ls.line_search_wolfe2(f, fp, x, p, amax=30, c2=c2)
    assert_line_wolfe(x, p, s, f, fp)
    s, _, _, _, _, _ = assert_warns(LineSearchWarning, ls.line_search_wolfe2, f, fp, x, p, amax=29, c2=c2)
    assert s is None
    assert_warns(LineSearchWarning, ls.line_search_wolfe2, f, fp, x, p, c2=c2, maxiter=5)