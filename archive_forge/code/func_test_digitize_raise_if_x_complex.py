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
def test_digitize_raise_if_x_complex(self):
    self.disable_leak_check()
    pyfunc = digitize
    cfunc = jit(nopython=True)(pyfunc)
    x = np.array([1 + 1j])
    y = np.array([1.0, 3.0, 4.5, 8.0])
    msg = 'x may not be complex'
    for func in (pyfunc, cfunc):
        with self.assertTypingError() as raises:
            func(x, y)
            self.assertIn(msg, str(raises.exception))