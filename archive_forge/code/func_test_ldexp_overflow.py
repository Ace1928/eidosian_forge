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
def test_ldexp_overflow(self):
    with np.errstate(over='ignore'):
        imax = np.iinfo(np.dtype('l')).max
        imin = np.iinfo(np.dtype('l')).min
        assert_equal(ncu.ldexp(2.0, imax), np.inf)
        assert_equal(ncu.ldexp(2.0, imin), 0)