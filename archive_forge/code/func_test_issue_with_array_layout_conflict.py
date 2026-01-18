import multiprocessing
import platform
import threading
import pickle
import weakref
from itertools import chain
from io import StringIO
import numpy as np
from numba import njit, jit, typeof, vectorize
from numba.core import types, errors
from numba import _dispatcher
from numba.tests.support import TestCase, captured_stdout
from numba.np.numpy_support import as_dtype
from numba.core.dispatcher import Dispatcher
from numba.extending import overload
from numba.tests.support import needs_lapack, SerialMixin
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
import unittest
def test_issue_with_array_layout_conflict(self):
    """
        This test an issue with the dispatcher when an array that is both
        C and F contiguous is supplied as the first signature.
        The dispatcher checks for F contiguous first but the compiler checks
        for C contiguous first. This results in an C contiguous code inserted
        as F contiguous function.
        """

    def pyfunc(A, i, j):
        return A[i, j]
    cfunc = jit(pyfunc)
    ary_c_and_f = np.array([[1.0]])
    ary_c = np.array([[0.0, 1.0], [2.0, 3.0]], order='C')
    ary_f = np.array([[0.0, 1.0], [2.0, 3.0]], order='F')
    exp_c = pyfunc(ary_c, 1, 0)
    exp_f = pyfunc(ary_f, 1, 0)
    self.assertEqual(1.0, cfunc(ary_c_and_f, 0, 0))
    got_c = cfunc(ary_c, 1, 0)
    got_f = cfunc(ary_f, 1, 0)
    self.assertEqual(exp_c, got_c)
    self.assertEqual(exp_f, got_f)