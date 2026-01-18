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
def test_disabled_compilation(self):

    @jit
    def foo(a):
        return a
    foo.compile('(float32,)')
    foo.disable_compile()
    with self.assertRaises(RuntimeError) as raises:
        foo.compile('(int32,)')
    self.assertEqual(str(raises.exception), 'compilation disabled')
    self.assertEqual(len(foo.signatures), 1)