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
def test_against_cmath(self):
    import cmath
    points = [-1 - 1j, -1 + 1j, +1 - 1j, +1 + 1j]
    name_map = {'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan', 'arcsinh': 'asinh', 'arccosh': 'acosh', 'arctanh': 'atanh'}
    atol = 4 * np.finfo(complex).eps
    for func in self.funcs:
        fname = func.__name__.split('.')[-1]
        cname = name_map.get(fname, fname)
        try:
            cfunc = getattr(cmath, cname)
        except AttributeError:
            continue
        for p in points:
            a = complex(func(np.complex_(p)))
            b = cfunc(p)
            assert_(abs(a - b) < atol, '%s %s: %s; cmath: %s' % (fname, p, a, b))