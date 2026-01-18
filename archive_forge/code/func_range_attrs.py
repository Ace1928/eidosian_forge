import unittest
import sys
import numpy
from numba import jit, njit
from numba.core import types, utils
from numba.tests.support import tag
from numba.cpython.rangeobj import length_of_iterator
def range_attrs(start, stop, step):
    r1 = range(start)
    r2 = range(start, stop)
    r3 = range(start, stop, step)
    tmp = []
    for r in (r1, r2, r3):
        tmp.append((r.start, r.stop, r.step))
    return tmp