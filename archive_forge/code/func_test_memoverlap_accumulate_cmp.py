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
@pytest.mark.parametrize('ufunc, dtype', [(ufunc, t[0]) for ufunc in UFUNCS_BINARY_ACC for t in ufunc.types if t[-1] == '?' and t[0] not in 'DFGMmO'])
def test_memoverlap_accumulate_cmp(ufunc, dtype):
    if ufunc.signature:
        pytest.skip('For generic signatures only')
    for size in (2, 8, 32, 64, 128, 256):
        arr = np.array([0, 1, 1] * size, dtype=dtype)
        acc = ufunc.accumulate(arr, dtype='?')
        acc_u8 = acc.view(np.uint8)
        exp = np.array(list(itertools.accumulate(arr, ufunc)), dtype=np.uint8)
        assert_equal(exp, acc_u8)