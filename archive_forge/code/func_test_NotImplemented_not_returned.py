import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def test_NotImplemented_not_returned(self):
    binary_funcs = [np.power, np.add, np.subtract, np.multiply, np.divide, np.true_divide, np.floor_divide, np.bitwise_and, np.bitwise_or, np.bitwise_xor, np.left_shift, np.right_shift, np.fmax, np.fmin, np.fmod, np.hypot, np.logaddexp, np.logaddexp2, np.maximum, np.minimum, np.mod, np.greater, np.greater_equal, np.less, np.less_equal, np.equal, np.not_equal]
    a = np.array('1')
    b = 1
    c = np.array([1.0, 2.0])
    for f in binary_funcs:
        assert_raises(TypeError, f, a, b)
        assert_raises(TypeError, f, c, a)