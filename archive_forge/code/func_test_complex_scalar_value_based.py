import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
@pytest.mark.parametrize(['other', 'expected'], [(np.bool_, np.complex128), (np.int64, np.complex128), (np.float16, np.complex64), (np.float32, np.complex64), (np.float64, np.complex128), (np.longdouble, np.clongdouble), (np.complex64, np.complex64), (np.complex128, np.complex128), (np.clongdouble, np.clongdouble)])
def test_complex_scalar_value_based(self, other, expected):
    complex_scalar = 1j
    res = np.result_type(other, complex_scalar)
    assert res == expected
    res = np.minimum(np.ones(3, dtype=other), complex_scalar).dtype
    assert res == expected