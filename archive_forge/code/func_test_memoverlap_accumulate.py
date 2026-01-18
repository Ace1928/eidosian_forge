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
@pytest.mark.parametrize('ftype', [np.float32, np.float64])
def test_memoverlap_accumulate(ftype):
    arr = np.array([0.61, 0.6, 0.77, 0.41, 0.19], dtype=ftype)
    out_max = np.array([0.61, 0.61, 0.77, 0.77, 0.77], dtype=ftype)
    out_min = np.array([0.61, 0.6, 0.6, 0.41, 0.19], dtype=ftype)
    assert_equal(np.maximum.accumulate(arr), out_max)
    assert_equal(np.minimum.accumulate(arr), out_min)