import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def test_inner_function_with_closure_3(self):
    code = '\n            def outer(x):\n                y = x + 1\n                z = 0\n\n                def inner(x):\n                    nonlocal z\n                    z += x * x\n                    return z + y\n\n                return inner(x) + inner(x) + z\n        '
    ns = {}
    exec(code.strip(), ns)
    cfunc = njit(ns['outer'])
    self.assertEqual(cfunc(10), ns['outer'](10))