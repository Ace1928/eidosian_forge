import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def enumerate_array_usecase():
    res = 0
    arrays = (np.ones(4), np.ones(5))
    for i, v in enumerate(arrays):
        res += v.sum()
    return res