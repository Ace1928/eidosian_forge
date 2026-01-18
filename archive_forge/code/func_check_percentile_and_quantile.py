from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_percentile_and_quantile(self, pyfunc, q_upper_bound):
    cfunc = jit(nopython=True)(pyfunc)

    def check(a, q, abs_tol=1e-12):
        expected = pyfunc(a, q)
        got = cfunc(a, q)
        finite = np.isfinite(expected)
        if np.all(finite):
            self.assertPreciseEqual(got, expected, abs_tol=abs_tol)
        else:
            self.assertPreciseEqual(got[finite], expected[finite], abs_tol=abs_tol)
    a = self.random.randn(27).reshape(3, 3, 3)
    q = np.linspace(0, q_upper_bound, 14)[::-1]
    check(a, q)
    check(a, 0)
    check(a, q_upper_bound / 2)
    check(a, q_upper_bound)
    not_finite = [np.nan, -np.inf, np.inf]
    a.flat[:10] = self.random.choice(not_finite, 10)
    self.random.shuffle(a)
    self.random.shuffle(q)
    check(a, q)
    a = a.flatten().tolist()
    q = q.flatten().tolist()
    check(a, q)
    check(tuple(a), tuple(q))
    a = self.random.choice([1, 2, 3, 4], 10)
    q = np.linspace(0, q_upper_bound, 5)
    check(a, q)
    x = np.arange(8) * 0.5
    np.testing.assert_equal(cfunc(x, 0), 0.0)
    np.testing.assert_equal(cfunc(x, q_upper_bound), 3.5)
    np.testing.assert_equal(cfunc(x, q_upper_bound / 2), 1.75)
    x = np.arange(12).reshape(3, 4)
    q = np.array((0.25, 0.5, 1.0)) * q_upper_bound
    np.testing.assert_equal(cfunc(x, q), [2.75, 5.5, 11.0])
    x = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    q = np.array((0.25, 0.5)) * q_upper_bound
    np.testing.assert_equal(cfunc(x, q).shape, (2,))
    q = np.array((0.25, 0.5, 0.75)) * q_upper_bound
    np.testing.assert_equal(cfunc(x, q).shape, (3,))
    x = np.arange(12).reshape(3, 4)
    np.testing.assert_equal(cfunc(x, q_upper_bound / 2), 5.5)
    self.assertTrue(np.isscalar(cfunc(x, q_upper_bound / 2)))
    np.testing.assert_equal(cfunc([1, 2, 3], 0), 1)
    a = np.array([2, 3, 4, 1])
    cfunc(a, [q_upper_bound / 2])
    np.testing.assert_equal(a, np.array([2, 3, 4, 1]))