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
def test_integer_power_of_1(self):
    dtypes = np.typecodes['AllInteger']
    for dt in dtypes:
        arr = np.arange(10, dtype=dt)
        assert_equal(np.power(1, arr), np.ones_like(arr))