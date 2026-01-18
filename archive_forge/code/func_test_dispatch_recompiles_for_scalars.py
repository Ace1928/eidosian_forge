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
def test_dispatch_recompiles_for_scalars(self):

    def foo(x):
        return x
    jitfoo = jit(nopython=True)(foo)
    jitfoo(np.complex128(1 + 2j))
    jitfoo(np.int32(10))
    jitfoo(np.bool_(False))
    self.assertEqual(len(jitfoo.signatures), 3)
    expected_sigs = [(types.complex128,), (types.int32,), (types.bool_,)]
    self.assertEqual(jitfoo.signatures, expected_sigs)
    jitfoo = jit([(types.complex128,)], nopython=True)(foo)
    jitfoo(np.complex128(1 + 2j))
    jitfoo(np.int32(10))
    jitfoo(np.bool_(False))
    self.assertEqual(len(jitfoo.signatures), 1)
    expected_sigs = [(types.complex128,)]
    self.assertEqual(jitfoo.signatures, expected_sigs)