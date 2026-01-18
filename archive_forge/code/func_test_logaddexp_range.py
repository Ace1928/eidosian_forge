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
def test_logaddexp_range(self):
    x = [1000000, -1000000, 1000200, -1000200]
    y = [1000200, -1000200, 1000000, -1000000]
    z = [1000200, -1000000, 1000200, -1000000]
    for dt in ['f', 'd', 'g']:
        logxf = np.array(x, dtype=dt)
        logyf = np.array(y, dtype=dt)
        logzf = np.array(z, dtype=dt)
        assert_almost_equal(np.logaddexp(logxf, logyf), logzf)