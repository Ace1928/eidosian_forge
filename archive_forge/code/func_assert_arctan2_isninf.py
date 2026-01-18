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
def assert_arctan2_isninf(x, y):
    assert_(np.isinf(ncu.arctan2(x, y)) and ncu.arctan2(x, y) < 0, 'arctan(%s, %s) is %s, not -inf' % (x, y, ncu.arctan2(x, y)))