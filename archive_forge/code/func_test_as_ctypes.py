import array
import cmath
from functools import reduce
import itertools
from operator import mul
import math
import symengine as se
from symengine.test_utilities import raises
from symengine import have_numpy
import unittest
from unittest.case import SkipTest
@unittest.skipUnless(have_numpy, 'Numpy not installed')
def test_as_ctypes():
    import numpy as np
    import ctypes
    x, y, z = se.symbols('x, y, z')
    l = se.Lambdify([x, y, z], [x + y + z, x * y * z + 1])
    addr1, addr2 = l.as_ctypes()
    inp = np.array([1, 2, 3], dtype=np.double)
    out = np.array([0, 0], dtype=np.double)
    addr1(out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), inp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), addr2)
    assert np.all(out == [6, 7])