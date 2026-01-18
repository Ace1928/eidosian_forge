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
def test_inspect_cfg(self):

    @jit
    def foo(the_array):
        return the_array.sum()
    a1 = np.ones(1)
    a2 = np.ones((1, 1))
    a3 = np.ones((1, 1, 1))
    foo(a1)
    foo(a2)
    foo(a3)
    cfgs = foo.inspect_cfg()
    self.assertEqual(len(cfgs), 3)
    [s1, s2, s3] = cfgs.keys()
    self.assertEqual(set([s1, s2, s3]), set(map(lambda x: (typeof(x),), [a1, a2, a3])))
    for cfg in cfgs.values():
        self._check_cfg_display(cfg)
    self.assertEqual(len(list(cfgs.values())), 3)
    cfg = foo.inspect_cfg(signature=foo.signatures[0])
    self._check_cfg_display(cfg)