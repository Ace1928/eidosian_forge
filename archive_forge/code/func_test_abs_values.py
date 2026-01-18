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
def test_abs_values(self):
    x = [np.nan, np.nan, np.inf, np.inf, 0.0, 0.0, 1.0, 1.0]
    y = [np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, -1.0, 1.0]
    for dt in ['e', 'f', 'd', 'g']:
        xf = np.array(x, dtype=dt)
        yf = np.array(y, dtype=dt)
        assert_equal(np.abs(yf), xf)