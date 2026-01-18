from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def bad_index(arr, arr2d):
    x = (arr.x,)
    y = arr.y
    arr2d[x, y] = 1.0