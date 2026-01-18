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
def test_integer_to_negative_power(self):
    dtypes = np.typecodes['Integer']
    for dt in dtypes:
        a = np.array([0, 1, 2, 3], dtype=dt)
        b = np.array([0, 1, 2, -3], dtype=dt)
        one = np.array(1, dtype=dt)
        minusone = np.array(-1, dtype=dt)
        assert_raises(ValueError, np.power, a, b)
        assert_raises(ValueError, np.power, a, minusone)
        assert_raises(ValueError, np.power, one, b)
        assert_raises(ValueError, np.power, one, minusone)