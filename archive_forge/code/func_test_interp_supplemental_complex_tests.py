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
def test_interp_supplemental_complex_tests(self):
    pyfunc = interp
    cfunc = jit(nopython=True)(pyfunc)
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5) + (1 + np.linspace(0, 1, 5)) * 1j
    x0 = 0.3
    y0 = x0 + (1 + x0) * 1j
    np.testing.assert_almost_equal(cfunc(x0, x, y), y0)