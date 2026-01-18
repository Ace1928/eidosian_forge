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
def test_cross1d(self):
    """Test with fixed-sized signature."""
    a = np.eye(3)
    assert_array_equal(umt.cross1d(a, a), np.zeros((3, 3)))
    out = np.zeros((3, 3))
    result = umt.cross1d(a[0], a, out)
    assert_(result is out)
    assert_array_equal(result, np.vstack((np.zeros(3), a[2], -a[1])))
    assert_raises(ValueError, umt.cross1d, np.eye(4), np.eye(4))
    assert_raises(ValueError, umt.cross1d, a, np.arange(4.0))
    assert_raises(ValueError, umt.cross1d, a, np.arange(3.0), np.zeros((3, 4)))
    assert_raises(ValueError, umt.cross1d, a, np.arange(3.0), np.zeros(3))