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
def test_incontiguous_array(self):
    msg = 'incontiguous memory layout of array'
    x = np.arange(64).reshape((2, 2, 2, 2, 2, 2))
    a = x[:, 0, :, 0, :, 0]
    b = x[:, 1, :, 1, :, 1]
    a[0, 0, 0] = -1
    msg2 = 'make sure it references to the original array'
    assert_equal(x[0, 0, 0, 0, 0, 0], -1, err_msg=msg2)
    assert_array_equal(umt.inner1d(a, b), np.sum(a * b, axis=-1), err_msg=msg)
    x = np.arange(24).reshape(2, 3, 4)
    a = x.T
    b = x.T
    a[0, 0, 0] = -1
    assert_equal(x[0, 0, 0], -1, err_msg=msg2)
    assert_array_equal(umt.inner1d(a, b), np.sum(a * b, axis=-1), err_msg=msg)