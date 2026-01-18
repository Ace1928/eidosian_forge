from numba import int32, int64
from numba import jit
from numba.core import types
from numba.extending import overload
from numba.tests.support import TestCase, tag
import unittest
@jit(nopython=True)
def g_inner(a, b=2, c=3):
    return (a, b, c)