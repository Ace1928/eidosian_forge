import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_ip_not_allclose_numpy(self):
    pyfunc = np_allclose
    cfunc = jit(nopython=True)(pyfunc)
    aran = np.arange(125).astype(dtype=np.float64).reshape((5, 5, 5))
    atol = 1e-08
    rtol = 1e-05
    numpy_data = [(np.asarray([np.inf, 0]), np.asarray([1.0, np.inf])), (np.asarray([np.inf, 0]), np.asarray([1.0, 0])), (np.asarray([np.inf, np.inf]), np.asarray([1.0, np.inf])), (np.asarray([np.inf, np.inf]), np.asarray([1.0, 0.0])), (np.asarray([-np.inf, 0.0]), np.asarray([np.inf, 0.0])), (np.asarray([np.nan, 0.0]), np.asarray([np.nan, 0.0])), (np.asarray([atol * 2]), np.asarray([0.0])), (np.asarray([1.0]), np.asarray([1 + rtol + atol * 2])), (aran, aran + aran * atol + atol * 2), (np.array([np.inf, 1.0]), np.array([0.0, np.inf]))]
    for x, y in numpy_data:
        self.assertEqual(pyfunc(x, y), cfunc(x, y))