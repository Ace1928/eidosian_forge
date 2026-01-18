import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
@pytest.mark.parametrize('dividend,divisor,quotient', [(np.timedelta64(2, 'Y'), np.timedelta64(2, 'M'), 12), (np.timedelta64(2, 'Y'), np.timedelta64(-2, 'M'), -12), (np.timedelta64(-2, 'Y'), np.timedelta64(2, 'M'), -12), (np.timedelta64(-2, 'Y'), np.timedelta64(-2, 'M'), 12), (np.timedelta64(2, 'M'), np.timedelta64(-2, 'Y'), -1), (np.timedelta64(2, 'Y'), np.timedelta64(0, 'M'), 0), (np.timedelta64(2, 'Y'), 2, np.timedelta64(1, 'Y')), (np.timedelta64(2, 'Y'), -2, np.timedelta64(-1, 'Y')), (np.timedelta64(-2, 'Y'), 2, np.timedelta64(-1, 'Y')), (np.timedelta64(-2, 'Y'), -2, np.timedelta64(1, 'Y')), (np.timedelta64(-2, 'Y'), -2, np.timedelta64(1, 'Y')), (np.timedelta64(-2, 'Y'), -3, np.timedelta64(0, 'Y')), (np.timedelta64(-2, 'Y'), 0, np.timedelta64('Nat', 'Y'))])
def test_division_int_timedelta(self, dividend, divisor, quotient):
    if divisor and (isinstance(quotient, int) or not np.isnat(quotient)):
        msg = 'Timedelta floor division check'
        assert dividend // divisor == quotient, msg
        msg = 'Timedelta arrays floor division check'
        dividend_array = np.array([dividend] * 5)
        quotient_array = np.array([quotient] * 5)
        assert all(dividend_array // divisor == quotient_array), msg
    else:
        if IS_WASM:
            pytest.skip("fp errors don't work in wasm")
        with np.errstate(divide='raise', invalid='raise'):
            with pytest.raises(FloatingPointError):
                dividend // divisor