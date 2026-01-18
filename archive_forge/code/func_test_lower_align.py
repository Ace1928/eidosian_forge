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
def test_lower_align(self):
    d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
    assert_equal(np.abs(d), d)
    assert_equal(np.negative(d), -d)
    np.negative(d, out=d)
    np.negative(np.ones_like(d), out=d)
    np.abs(d, out=d)
    np.abs(np.ones_like(d), out=d)