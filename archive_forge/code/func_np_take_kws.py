import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
def np_take_kws(A, indices, axis):
    return np.take(A, indices, axis=axis)