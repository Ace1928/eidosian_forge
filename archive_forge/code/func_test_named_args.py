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
def test_named_args(self):
    """
        Test passing named arguments to a dispatcher.
        """
    f, check = self.compile_func(addsub)
    check(3, z=10, y=4)
    check(3, 4, 10)
    check(x=3, y=4, z=10)
    self.assertEqual(len(f.overloads), 1)
    with self.assertRaises(TypeError) as cm:
        f(3, 4, y=6, z=7)
    self.assertIn('too many arguments: expected 3, got 4', str(cm.exception))
    with self.assertRaises(TypeError) as cm:
        f()
    self.assertIn('not enough arguments: expected 3, got 0', str(cm.exception))
    with self.assertRaises(TypeError) as cm:
        f(3, 4, y=6)
    self.assertIn("missing argument 'z'", str(cm.exception))