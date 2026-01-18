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
@pytest.mark.parametrize('dtype', ('e', 'f', 'd'))
def test_divide_spurious_fpexception(self, dtype):
    dt = np.dtype(dtype)
    dt_info = np.finfo(dt)
    subnorm = dt_info.smallest_subnormal
    with assert_no_warnings():
        np.zeros(128 + 1, dtype=dt) / subnorm