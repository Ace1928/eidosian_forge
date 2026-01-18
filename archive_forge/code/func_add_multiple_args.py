import numpy as np
from numba import float32, jit, njit
from numba.np.ufunc import Vectorize
from numba.core.errors import TypingError
from numba.tests.support import TestCase
import unittest
def add_multiple_args(a, b, c, d):
    return a + b + c + d