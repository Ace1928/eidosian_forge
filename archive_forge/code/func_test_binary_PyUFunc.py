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
@pytest.mark.parametrize('input_dtype,output_dtype', np_dtypes)
def test_binary_PyUFunc(self, input_dtype, output_dtype, f=f2, x=0, y=1):
    xs = np.full(10, input_dtype(x), dtype=output_dtype)
    ys = f(xs, xs)[::2]
    assert_allclose(ys, y)
    assert_equal(ys.dtype, output_dtype)