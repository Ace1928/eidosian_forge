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
def test_casting_out_param(self):
    a = np.ones((200, 100), np.int64)
    b = np.ones((200, 100), np.int64)
    c = np.ones((200, 100), np.float64)
    np.add(a, b, out=c)
    assert_equal(c, 2)
    a = np.zeros(65536)
    b = np.zeros(65536, dtype=np.float32)
    np.subtract(a, 0, out=b)
    assert_equal(b, 0)