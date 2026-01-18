from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def np_frombuffer_allocated(shape):
    """
    np.frombuffer() on a Numba-allocated buffer.
    """
    arr = np.ones(shape, dtype=np.int32)
    return np.frombuffer(arr)