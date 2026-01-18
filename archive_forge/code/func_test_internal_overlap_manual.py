import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def test_internal_overlap_manual():
    x = np.arange(1).astype(np.int8)
    check_internal_overlap(x, False)
    check_internal_overlap(x.reshape([]), False)
    a = as_strided(x, strides=(3, 4), shape=(4, 4))
    check_internal_overlap(a, False)
    a = as_strided(x, strides=(3, 4), shape=(5, 4))
    check_internal_overlap(a, True)
    a = as_strided(x, strides=(0,), shape=(0,))
    check_internal_overlap(a, False)
    a = as_strided(x, strides=(0,), shape=(1,))
    check_internal_overlap(a, False)
    a = as_strided(x, strides=(0,), shape=(2,))
    check_internal_overlap(a, True)
    a = as_strided(x, strides=(0, -9993), shape=(87, 22))
    check_internal_overlap(a, True)
    a = as_strided(x, strides=(0, -9993), shape=(1, 22))
    check_internal_overlap(a, False)
    a = as_strided(x, strides=(0, -9993), shape=(0, 22))
    check_internal_overlap(a, False)