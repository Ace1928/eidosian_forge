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
def test_interp_supplemental_tests(self):
    pyfunc = interp
    cfunc = jit(nopython=True)(pyfunc)
    for size in range(1, 10):
        xp = np.arange(size, dtype=np.double)
        yp = np.ones(size, dtype=np.double)
        incpts = np.array([-1, 0, size - 1, size], dtype=np.double)
        decpts = incpts[::-1]
        incres = cfunc(incpts, xp, yp)
        decres = cfunc(decpts, xp, yp)
        inctgt = np.array([1, 1, 1, 1], dtype=float)
        dectgt = inctgt[::-1]
        np.testing.assert_almost_equal(incres, inctgt)
        np.testing.assert_almost_equal(decres, dectgt)
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    x0 = 0
    np.testing.assert_almost_equal(cfunc(x0, x, y), x0)
    x0 = 0.3
    np.testing.assert_almost_equal(cfunc(x0, x, y), x0)
    x0 = np.float32(0.3)
    np.testing.assert_almost_equal(cfunc(x0, x, y), x0)
    x0 = np.float64(0.3)
    np.testing.assert_almost_equal(cfunc(x0, x, y), x0)
    x0 = np.nan
    np.testing.assert_almost_equal(cfunc(x0, x, y), x0)
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    x0 = np.array(0.3)
    np.testing.assert_almost_equal(cfunc(x0, x, y), x0)
    xp = np.arange(0, 10, 0.0001)
    fp = np.sin(xp)
    np.testing.assert_almost_equal(cfunc(np.pi, xp, fp), 0.0)