from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
@staticmethod
def array_like_variations():
    yield ((1.1, 2.2), (3.3, 4.4), (5.5, 6.6))
    yield (0.0, 1.0, 0.0, -6.0)
    yield ([0, 1], [2, 3])
    yield ()
    yield np.nan
    yield 0
    yield 1
    yield False
    yield True
    yield (True, False, True)
    yield (2 + 1j)
    yield None
    yield 'a_string'
    yield ''