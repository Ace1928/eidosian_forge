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
@requires_memory(6 * 1024 ** 3)
@pytest.mark.skipif(sys.maxsize < 2 ** 32, reason='test array too large for 32bit platform')
def test_identityless_reduction_huge_array(self):
    arr = np.zeros((2, 2 ** 31), 'uint8')
    arr[:, 0] = [1, 3]
    arr[:, -1] = [4, 1]
    res = np.maximum.reduce(arr, axis=0)
    del arr
    assert res[0] == 3
    assert res[-1] == 4