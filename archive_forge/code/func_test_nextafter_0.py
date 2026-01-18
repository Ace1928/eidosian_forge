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
def test_nextafter_0():
    for t, direction in itertools.product(np.sctypes['float'], (1, -1)):
        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            if not np.isnan(np.finfo(t).tiny):
                tiny = np.finfo(t).tiny
                assert_(0.0 < direction * np.nextafter(t(0), t(direction)) < tiny)
        assert_equal(np.nextafter(t(0), t(direction)) / t(2.1), direction * 0.0)