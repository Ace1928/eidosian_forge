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
def test_fingerprint_failure(self):
    """
        Failure in computing the fingerprint cannot affect a nopython=False
        function.  On the other hand, with nopython=True, a ValueError should
        be raised to report the failure with fingerprint.
        """

    def foo(x):
        return x
    errmsg = 'cannot compute fingerprint of empty list'
    with self.assertRaises(ValueError) as raises:
        _dispatcher.compute_fingerprint([])
    self.assertIn(errmsg, str(raises.exception))
    objmode_foo = jit(forceobj=True)(foo)
    self.assertEqual(objmode_foo([]), [])
    strict_foo = jit(nopython=True)(foo)
    with self.assertRaises(ValueError) as raises:
        strict_foo([])
    self.assertIn(errmsg, str(raises.exception))

    @jit(forceobj=True)
    def bar():
        object()
        x = []
        for i in range(10):
            x = objmode_foo(x)
        return x
    self.assertEqual(bar(), [])
    [cr] = bar.overloads.values()
    self.assertEqual(len(cr.lifted), 1)