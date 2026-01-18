from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def test_issue_812(self):
    from numba import jit

    @jit('f8[:](f8[:])', forceobj=True)
    def test(x):
        res = np.zeros(len(x))
        ind = 0
        for ii in range(len(x)):
            ind += 1
            res[ind] = x[ind]
            if x[ind] >= 10:
                break
        for ii in range(ind + 1, len(x)):
            res[ii] = 0
        return res
    x = np.array([1.0, 4, 2, -3, 5, 2, 10, 5, 2, 6])
    np.testing.assert_equal(test.py_func(x), test(x))