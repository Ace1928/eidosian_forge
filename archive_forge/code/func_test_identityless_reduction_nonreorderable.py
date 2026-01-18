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
def test_identityless_reduction_nonreorderable(self):
    a = np.array([[8.0, 2.0, 2.0], [1.0, 0.5, 0.25]])
    res = np.divide.reduce(a, axis=0)
    assert_equal(res, [8.0, 4.0, 8.0])
    res = np.divide.reduce(a, axis=1)
    assert_equal(res, [2.0, 8.0])
    res = np.divide.reduce(a, axis=())
    assert_equal(res, a)
    assert_raises(ValueError, np.divide.reduce, a, axis=(0, 1))