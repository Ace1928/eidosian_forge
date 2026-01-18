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
def test_division_complex(self):
    msg = 'Complex division implementation check'
    x = np.array([1.0 + 1.0 * 1j, 1.0 + 0.5 * 1j, 1.0 + 2.0 * 1j], dtype=np.complex128)
    assert_almost_equal(x ** 2 / x, x, err_msg=msg)
    msg = 'Complex division overflow/underflow check'
    x = np.array([1e+110, 1e-110], dtype=np.complex128)
    y = x ** 2 / x
    assert_almost_equal(y / x, [1, 1], err_msg=msg)