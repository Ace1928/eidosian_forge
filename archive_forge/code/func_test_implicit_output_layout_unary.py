import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
def test_implicit_output_layout_unary(self):

    def pyfunc(a0):
        return np.sqrt(a0)
    X = np.linspace(0, 1, 20).reshape(4, 5)
    Y = np.array(X, order='F')
    Z = X.reshape(5, 4).T[0]
    Xty = typeof(X)
    assert X.flags.c_contiguous and Xty.layout == 'C'
    Yty = typeof(Y)
    assert Y.flags.f_contiguous and Yty.layout == 'F'
    Zty = typeof(Z)
    assert Zty.layout == 'A'
    assert not Z.flags.c_contiguous
    assert not Z.flags.f_contiguous
    for arg0 in [X, Y, Z]:
        args = (typeof(arg0),)
        cfunc = self._compile(pyfunc, args, nrt=True)
        expected = pyfunc(arg0)
        result = cfunc(arg0)
        self.assertEqual(expected.flags.c_contiguous, result.flags.c_contiguous)
        self.assertEqual(expected.flags.f_contiguous, result.flags.f_contiguous)
        np.testing.assert_array_equal(expected, result)