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
def test_integer_power(self):
    a = np.array([15, 15], 'i8')
    b = np.power(a, a)
    assert_equal(b, [437893890380859375, 437893890380859375])