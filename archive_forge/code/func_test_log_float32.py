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
def test_log_float32(self):
    np.random.seed(42)
    x_f32 = np.float32(np.random.uniform(low=0.0, high=1000, size=1000000))
    x_f64 = np.float64(x_f32)
    assert_array_max_ulp(np.log(x_f32), np.float32(np.log(x_f64)), maxulp=4)