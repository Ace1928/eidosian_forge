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
def test_endian(self):
    msg = 'big endian'
    a = np.arange(6, dtype='>i4').reshape((2, 3))
    assert_array_equal(umt.inner1d(a, a), np.sum(a * a, axis=-1), err_msg=msg)
    msg = 'little endian'
    a = np.arange(6, dtype='<i4').reshape((2, 3))
    assert_array_equal(umt.inner1d(a, a), np.sum(a * a, axis=-1), err_msg=msg)
    Ba = np.arange(1, dtype='>f8')
    La = np.arange(1, dtype='<f8')
    assert_equal((Ba + Ba).dtype, np.dtype('f8'))
    assert_equal((Ba + La).dtype, np.dtype('f8'))
    assert_equal((La + Ba).dtype, np.dtype('f8'))
    assert_equal((La + La).dtype, np.dtype('f8'))
    assert_equal(np.absolute(La).dtype, np.dtype('f8'))
    assert_equal(np.absolute(Ba).dtype, np.dtype('f8'))
    assert_equal(np.negative(La).dtype, np.dtype('f8'))
    assert_equal(np.negative(Ba).dtype, np.dtype('f8'))