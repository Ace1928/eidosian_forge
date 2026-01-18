from numba import int32, int64
from numba import jit
from numba.core import types
from numba.extending import overload
from numba.tests.support import TestCase, tag
import unittest
def star_call(x, y, z):
    return (star_inner(x, *y), star_inner(*z))