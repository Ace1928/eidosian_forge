from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def check_all_axes(arr):
    for axis in range(-arr.ndim - 1, arr.ndim + 1):
        check(arr, axis)