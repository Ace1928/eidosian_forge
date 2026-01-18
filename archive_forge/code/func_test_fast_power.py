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
def test_fast_power(self):
    x = np.array([1, 2, 3], np.int16)
    res = x ** 2.0
    assert_((x ** 2.00001).dtype is res.dtype)
    assert_array_equal(res, [1, 4, 9])
    assert_(not np.may_share_memory(res, x))
    assert_array_equal(x, [1, 2, 3])
    res = x ** np.array([[[2]]])
    assert_equal(res.shape, (1, 1, 3))