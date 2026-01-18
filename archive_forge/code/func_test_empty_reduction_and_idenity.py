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
def test_empty_reduction_and_idenity(self):
    arr = np.zeros((0, 5))
    assert np.true_divide.reduce(arr, axis=1).shape == (0,)
    with pytest.raises(ValueError):
        np.true_divide.reduce(arr, axis=0)
    arr = np.zeros((0, 0, 5))
    with pytest.raises(ValueError):
        np.true_divide.reduce(arr, axis=1)
    res = np.true_divide.reduce(arr, axis=1, initial=1)
    assert_array_equal(res, np.ones((0, 5)))