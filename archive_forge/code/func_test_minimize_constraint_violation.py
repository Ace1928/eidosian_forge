import math
import numpy as np
from numpy.testing import assert_allclose, assert_, assert_array_equal
from scipy.optimize import fmin_cobyla, minimize, Bounds
def test_minimize_constraint_violation(self):
    np.random.seed(1234)
    pb = np.random.rand(10, 10)
    spread = np.random.rand(10)

    def p(w):
        return pb.dot(w)

    def f(w):
        return -(w * spread).sum()

    def c1(w):
        return 500 - abs(p(w)).sum()

    def c2(w):
        return 5 - abs(p(w).sum())

    def c3(w):
        return 5 - abs(p(w)).max()
    cons = ({'type': 'ineq', 'fun': c1}, {'type': 'ineq', 'fun': c2}, {'type': 'ineq', 'fun': c3})
    w0 = np.zeros((10,))
    sol = minimize(f, w0, method='cobyla', constraints=cons, options={'catol': 1e-06})
    assert_(sol.maxcv > 1e-06)
    assert_(not sol.success)