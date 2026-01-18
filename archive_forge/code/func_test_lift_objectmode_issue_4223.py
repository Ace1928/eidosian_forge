from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def test_lift_objectmode_issue_4223(self):
    from numba import jit

    @jit(forceobj=True)
    def foo(a, b, c, d, x0, y0, n):
        xs, ys = (np.zeros(n), np.zeros(n))
        xs[0], ys[0] = (x0, y0)
        for i in np.arange(n - 1):
            xs[i + 1] = np.sin(a * ys[i]) + c * np.cos(a * xs[i])
            ys[i + 1] = np.sin(b * xs[i]) + d * np.cos(b * ys[i])
        object()
        return (xs, ys)
    kwargs = dict(a=1.7, b=1.7, c=0.6, d=1.2, x0=0, y0=0, n=200)
    got = foo(**kwargs)
    expected = foo.py_func(**kwargs)
    self.assertPreciseEqual(got[0], expected[0])
    self.assertPreciseEqual(got[1], expected[1])
    [lifted] = foo.overloads[foo.signatures[0]].lifted
    self.assertEqual(len(lifted.nopython_signatures), 1)