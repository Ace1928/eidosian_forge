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
def test_innerwt_empty(self):
    """Test generalized ufunc with zero-sized operands"""
    a = np.array([], dtype='f8')
    b = np.array([], dtype='f8')
    w = np.array([], dtype='f8')
    assert_array_equal(umt.innerwt(a, b, w), np.sum(a * b * w, axis=-1))