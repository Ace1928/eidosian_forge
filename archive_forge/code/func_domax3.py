from numba import njit
from numba.core import types
import unittest
def domax3(a, b, c):
    return max(a, b, c)