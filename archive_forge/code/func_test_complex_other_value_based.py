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
@np._no_nep50_warning()
@pytest.mark.parametrize(['other', 'expected', 'expected_weak'], [(2 ** 16 - 1, np.complex64, None), (2 ** 32 - 1, np.complex128, np.complex64), (np.float16(2), np.complex64, None), (np.float32(2), np.complex64, None), (np.longdouble(2), np.complex64, np.clongdouble), (np.longdouble(np.nextafter(1.7e+308, 0.0)), np.complex128, np.clongdouble), (np.longdouble(np.nextafter(1.7e+308, np.inf)), np.clongdouble, None), (np.complex64(2), np.complex64, None), (np.clongdouble(2), np.complex64, np.clongdouble), (np.clongdouble(np.nextafter(1.7e+308, 0.0) * 1j), np.complex128, np.clongdouble), (np.clongdouble(np.nextafter(1.7e+308, np.inf)), np.clongdouble, None)])
def test_complex_other_value_based(self, weak_promotion, other, expected, expected_weak):
    if weak_promotion and expected_weak is not None:
        expected = expected_weak
    min_complex = np.dtype(np.complex64)
    res = np.result_type(other, min_complex)
    assert res == expected
    res = np.minimum(other, np.ones(3, dtype=min_complex)).dtype
    assert res == expected