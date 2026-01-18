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
def test_builtin_long(self):
    assert_equal(np.array(2 ** 200).item(), 2 ** 200)
    a = np.array(2 ** 100 * 3 ** 5)
    b = np.array([2 ** 100 * 5 ** 7, 2 ** 50 * 3 ** 10])
    assert_equal(np.gcd(a, b), [2 ** 100, 2 ** 50 * 3 ** 5])
    assert_equal(np.lcm(a, b), [2 ** 100 * 3 ** 5 * 5 ** 7, 2 ** 100 * 3 ** 10])
    assert_equal(np.gcd(2 ** 100, 3 ** 100), 1)