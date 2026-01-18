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
def test_dispatcher_raises_for_invalid_decoration(self):

    @jit(nopython=True)
    def foo(x):
        return x
    with self.assertRaises(TypeError) as raises:
        jit(foo)
    err_msg = str(raises.exception)
    self.assertIn('A jit decorator was called on an already jitted function', err_msg)
    self.assertIn('foo', err_msg)
    self.assertIn('.py_func', err_msg)
    with self.assertRaises(TypeError) as raises:
        jit(BaseTest)
    err_msg = str(raises.exception)
    self.assertIn('The decorated object is not a function', err_msg)
    self.assertIn(f'{type(BaseTest)}', err_msg)