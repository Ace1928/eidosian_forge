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
def test_finfo(self):
    types = [np.float32, np.float64, np.complex64, np.complex128]
    attrs = ('eps', 'epsneg', 'iexp', 'machep', 'max', 'maxexp', 'negep', 'nexp', 'nmant', 'precision', 'resolution', 'tiny', 'bits')
    for ty in types:
        self.check(finfo, attrs, ty(1))
        hc_func = self.create_harcoded_variant(np.finfo, ty)
        self.check(hc_func, attrs)
    with self.assertRaises(TypingError) as raises:
        cfunc = jit(nopython=True)(finfo_machar)
        cfunc(7.0)
    msg = "Unknown attribute 'machar' of type finfo"
    self.assertIn(msg, str(raises.exception))
    with self.assertTypingError():
        cfunc = jit(nopython=True)(finfo)
        cfunc(np.int32(7))