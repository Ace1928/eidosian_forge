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
def run_fc_multiproc(self, fc):
    try:
        ctx = multiprocessing.get_context('spawn')
    except AttributeError:
        ctx = multiprocessing
    for a in [1, 2, 3]:
        p = ctx.Process(target=_checker, args=(fc, a))
        p.start()
        p.join(_TEST_TIMEOUT)
        self.assertEqual(p.exitcode, 0)