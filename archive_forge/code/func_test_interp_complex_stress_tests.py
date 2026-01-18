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
def test_interp_complex_stress_tests(self):
    pyfunc = interp
    cfunc = jit(nopython=True)(pyfunc)
    ndata = 2000
    xp = np.linspace(0, 10, 1 + ndata)
    real = np.sin(xp / 2.0)
    real[:200] = self.rnd.choice([np.inf, -np.inf, np.nan], 200)
    self.rnd.shuffle(real)
    imag = np.cos(xp / 2.0)
    imag[:200] = self.rnd.choice([np.inf, -np.inf, np.nan], 200)
    self.rnd.shuffle(imag)
    fp = real + 1j * imag
    for x in self.arrays(ndata):
        expected = pyfunc(x, xp, fp)
        got = cfunc(x, xp, fp)
        np.testing.assert_allclose(expected, got, equal_nan=True)
        self.rnd.shuffle(x)
        self.rnd.shuffle(xp)
        self.rnd.shuffle(fp)
        np.testing.assert_allclose(expected, got, equal_nan=True)