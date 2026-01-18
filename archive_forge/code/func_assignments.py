import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def assignments(a):
    b = c = str(a)
    return b + c