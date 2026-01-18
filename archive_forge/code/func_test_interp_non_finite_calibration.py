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
def test_interp_non_finite_calibration(self):
    pyfunc = interp
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc)
    xp = np.array([0, 1, 9, 10])
    fp = np.array([-np.inf, 0.1, 0.9, np.inf])
    x = np.array([0.2, 9.5])
    params = {'x': x, 'xp': xp, 'fp': fp}
    _check(params)
    xp = np.array([-np.inf, 1, 9, np.inf])
    fp = np.array([0, 0.1, 0.9, 1])
    x = np.array([0.2, 9.5])
    params = {'x': x, 'xp': xp, 'fp': fp}
    _check(params)