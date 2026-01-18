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
def test_axis_out_of_bounds(self):
    a = np.array([False, False])
    assert_raises(np.AxisError, a.all, axis=1)
    a = np.array([False, False])
    assert_raises(np.AxisError, a.all, axis=-2)
    a = np.array([False, False])
    assert_raises(np.AxisError, a.any, axis=1)
    a = np.array([False, False])
    assert_raises(np.AxisError, a.any, axis=-2)