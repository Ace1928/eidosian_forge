import numpy as np
from numba import float32, jit, njit
from numba.np.ufunc import Vectorize
from numba.core.errors import TypingError
from numba.tests.support import TestCase
import unittest
def gufunc_add(a, b):
    result = 0.0
    for i in range(a.shape[0]):
        result += a[i] * b[i]
    return result