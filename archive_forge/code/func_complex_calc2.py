import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def complex_calc2(a, b):
    z = complex(a, b)
    return z.real ** 2 + z.imag ** 2