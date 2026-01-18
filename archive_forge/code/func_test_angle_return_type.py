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
def test_angle_return_type(self):

    def numba_angle(x):
        r = np.angle(x)
        return r.dtype
    pyfunc = numba_angle
    x_values = [1.0, -1.0, 1.0 + 0j, -5 - 5j]
    x_types = ['f4', 'f8', 'c8', 'c16']
    for val, typ in zip(x_values, x_types):
        x = np.array([val], dtype=typ)
        cfunc = jit(nopython=True)(pyfunc)
        expected = pyfunc(x)
        got = cfunc(x)
        self.assertEqual(expected, got)