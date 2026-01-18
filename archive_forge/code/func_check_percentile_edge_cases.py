from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_percentile_edge_cases(self, pyfunc, q_upper_bound=100):
    cfunc = jit(nopython=True)(pyfunc)

    def check(a, q, abs_tol=1e-14):
        expected = pyfunc(a, q)
        got = cfunc(a, q)
        finite = np.isfinite(expected)
        if np.all(finite):
            self.assertPreciseEqual(got, expected, abs_tol=abs_tol)
        else:
            self.assertPreciseEqual(got[finite], expected[finite], abs_tol=abs_tol)

    def convert_to_float_and_check(a, q, abs_tol=1e-14):
        expected = pyfunc(a, q).astype(np.float64)
        got = cfunc(a, q)
        self.assertPreciseEqual(got, expected, abs_tol=abs_tol)

    def _array_combinations(elements):
        for i in range(1, 10):
            for comb in combinations_with_replacement(elements, i):
                yield np.array(comb)
    q = (0, 0.1 * q_upper_bound, 0.2 * q_upper_bound, q_upper_bound)
    element_pool = (1, -1, np.nan, np.inf, -np.inf)
    for a in _array_combinations(element_pool):
        check(a, q)
    if q_upper_bound == 1:
        _check = convert_to_float_and_check
    else:
        _check = check
    a = np.array(5)
    q = np.array(1)
    _check(a, q)
    a = 5
    q = q_upper_bound / 2
    _check(a, q)