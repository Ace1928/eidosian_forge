import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
@jit(nopython=True)
def setitem_usecase(buf, i, v):
    buf[i] = v