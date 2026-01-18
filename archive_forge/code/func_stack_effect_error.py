import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def stack_effect_error(x):
    i = 2
    c = 1
    if i == x:
        for i in range(3):
            c = i
    return i + c