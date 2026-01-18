import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def test_Polynomial_constructor(self):

    def pyfunc3(c, dom, win):
        p = poly.Polynomial(c, dom, win)
        return p
    cfunc3 = njit(pyfunc3)

    def pyfunc1(c):
        p = poly.Polynomial(c)
        return p
    cfunc1 = njit(pyfunc1)
    list1 = (np.array([0, 1]), np.array([0.0, 1.0]))
    list2 = (np.array([0, 1]), np.array([0.0, 1.0]))
    list3 = (np.array([0, 1]), np.array([0.0, 1.0]))
    for c in list1:
        for dom in list2:
            for win in list3:
                p1 = pyfunc3(c, dom, win)
                p2 = cfunc3(c, dom, win)
                q1 = pyfunc1(c)
                q2 = cfunc1(c)
                self.assertPreciseEqual(p1, p2)
                self.assertPreciseEqual(p1.coef, p2.coef)
                self.assertPreciseEqual(p1.domain, p2.domain)
                self.assertPreciseEqual(p1.window, p2.window)
                self.assertPreciseEqual(q1.coef, q2.coef)
                self.assertPreciseEqual(q1.domain, q2.domain)
                self.assertPreciseEqual(q1.window, q2.window)