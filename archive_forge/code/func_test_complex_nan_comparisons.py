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
def test_complex_nan_comparisons():
    nans = [complex(np.nan, 0), complex(0, np.nan), complex(np.nan, np.nan)]
    fins = [complex(1, 0), complex(-1, 0), complex(0, 1), complex(0, -1), complex(1, 1), complex(-1, -1), complex(0, 0)]
    with np.errstate(invalid='ignore'):
        for x in nans + fins:
            x = np.array([x])
            for y in nans + fins:
                y = np.array([y])
                if np.isfinite(x) and np.isfinite(y):
                    continue
                assert_equal(x < y, False, err_msg='%r < %r' % (x, y))
                assert_equal(x > y, False, err_msg='%r > %r' % (x, y))
                assert_equal(x <= y, False, err_msg='%r <= %r' % (x, y))
                assert_equal(x >= y, False, err_msg='%r >= %r' % (x, y))
                assert_equal(x == y, False, err_msg='%r == %r' % (x, y))