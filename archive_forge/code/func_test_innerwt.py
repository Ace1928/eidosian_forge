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
def test_innerwt(self):
    a = np.arange(6).reshape((2, 3))
    b = np.arange(10, 16).reshape((2, 3))
    w = np.arange(20, 26).reshape((2, 3))
    assert_array_equal(umt.innerwt(a, b, w), np.sum(a * b * w, axis=-1))
    a = np.arange(100, 124).reshape((2, 3, 4))
    b = np.arange(200, 224).reshape((2, 3, 4))
    w = np.arange(300, 324).reshape((2, 3, 4))
    assert_array_equal(umt.innerwt(a, b, w), np.sum(a * b * w, axis=-1))