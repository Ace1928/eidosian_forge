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
def test_floor_division_signed_zero(self):
    x = np.zeros(10)
    assert_equal(np.signbit(x // 1), 0)
    assert_equal(np.signbit(-x // 1), 1)