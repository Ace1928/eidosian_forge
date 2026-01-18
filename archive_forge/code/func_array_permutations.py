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
def array_permutations():
    x = np.arange(9).reshape(3, 3)
    yield x
    yield (x * 1.1)
    yield np.asfortranarray(x)
    yield x[::-1]
    yield (np.linspace(-10, 10, 60).reshape(3, 4, 5) * 1j)