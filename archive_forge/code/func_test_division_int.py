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
def test_division_int(self):
    x = np.array([5, 10, 90, 100, -5, -10, -90, -100, -120])
    if 5 / 10 == 0.5:
        assert_equal(x / 100, [0.05, 0.1, 0.9, 1, -0.05, -0.1, -0.9, -1, -1.2])
    else:
        assert_equal(x / 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
    assert_equal(x // 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
    assert_equal(x % 100, [5, 10, 90, 0, 95, 90, 10, 0, 80])