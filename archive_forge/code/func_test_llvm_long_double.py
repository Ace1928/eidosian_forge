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
@unittest.skipUnless(se.have_llvm, 'No LLVM support')
@unittest.skipUnless(se.have_llvm_long_double, 'No LLVM IEEE-80 bit support')
def test_llvm_long_double():
    import numpy as np
    import ctypes
    from symengine.lib.symengine_wrapper import LLVMLongDouble
    x, y, z = se.symbols('x, y, z')
    l = se.Lambdify([x, y, z], [2 * x, y / z], dtype=np.longdouble, backend='llvm')
    inp = np.array([1, 2, 3], dtype=np.longdouble)
    exp_out = np.array([2, 2.0 / 3.0], dtype=np.longdouble)
    out = l(inp)
    assert type(l) == LLVMLongDouble
    assert out.dtype == np.longdouble
    assert np.allclose(out, exp_out)