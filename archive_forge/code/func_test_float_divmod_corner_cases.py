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
def test_float_divmod_corner_cases(self):
    for dt in np.typecodes['Float']:
        fnan = np.array(np.nan, dtype=dt)
        fone = np.array(1.0, dtype=dt)
        fzer = np.array(0.0, dtype=dt)
        finf = np.array(np.inf, dtype=dt)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in divmod')
            sup.filter(RuntimeWarning, 'divide by zero encountered in divmod')
            div, rem = np.divmod(fone, fzer)
            assert np.isinf(div), 'dt: %s, div: %s' % (dt, rem)
            assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
            div, rem = np.divmod(fzer, fzer)
            assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
            (assert_(np.isnan(div)), 'dt: %s, rem: %s' % (dt, rem))
            div, rem = np.divmod(finf, finf)
            assert np.isnan(div), 'dt: %s, rem: %s' % (dt, rem)
            assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
            div, rem = np.divmod(finf, fzer)
            assert np.isinf(div), 'dt: %s, rem: %s' % (dt, rem)
            assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
            div, rem = np.divmod(fnan, fone)
            assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
            assert np.isnan(div), 'dt: %s, rem: %s' % (dt, rem)
            div, rem = np.divmod(fone, fnan)
            assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
            assert np.isnan(div), 'dt: %s, rem: %s' % (dt, rem)
            div, rem = np.divmod(fnan, fzer)
            assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
            assert np.isnan(div), 'dt: %s, rem: %s' % (dt, rem)