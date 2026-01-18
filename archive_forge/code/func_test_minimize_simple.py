import math
import numpy as np
from numpy.testing import assert_allclose, assert_, assert_array_equal
from scipy.optimize import fmin_cobyla, minimize, Bounds
def test_minimize_simple(self):

    class Callback:

        def __init__(self):
            self.n_calls = 0
            self.last_x = None

        def __call__(self, x):
            self.n_calls += 1
            self.last_x = x
    callback = Callback()
    cons = ({'type': 'ineq', 'fun': self.con1}, {'type': 'ineq', 'fun': self.con2})
    sol = minimize(self.fun, self.x0, method='cobyla', constraints=cons, callback=callback, options=self.opts)
    assert_allclose(sol.x, self.solution, atol=0.0001)
    assert_(sol.success, sol.message)
    assert_(sol.maxcv < 1e-05, sol)
    assert_(sol.nfev < 70, sol)
    assert_(sol.fun < self.fun(self.solution) + 0.001, sol)
    assert_(sol.nfev == callback.n_calls, 'Callback is not called exactly once for every function eval.')
    assert_array_equal(sol.x, callback.last_x, 'Last design vector sent to the callback is not equal to returned value.')