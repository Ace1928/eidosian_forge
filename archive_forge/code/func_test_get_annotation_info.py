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
def test_get_annotation_info(self):

    @jit
    def foo(a):
        return a + 1
    foo(1)
    foo(1.3)
    expected = dict(chain.from_iterable((foo.get_annotation_info(i).items() for i in foo.signatures)))
    result = foo.get_annotation_info()
    self.assertEqual(expected, result)