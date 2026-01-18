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
def test_float_remainder_exact(self):
    nlst = list(range(-127, 0))
    plst = list(range(1, 128))
    dividend = nlst + [0] + plst
    divisor = nlst + plst
    arg = list(itertools.product(dividend, divisor))
    tgt = list((divmod(*t) for t in arg))
    a, b = np.array(arg, dtype=int).T
    tgtdiv, tgtrem = np.array(tgt, dtype=float).T
    tgtdiv = np.where((tgtdiv == 0.0) & ((b < 0) ^ (a < 0)), -0.0, tgtdiv)
    tgtrem = np.where((tgtrem == 0.0) & (b < 0), -0.0, tgtrem)
    for op in [floor_divide_and_remainder, np.divmod]:
        for dt in np.typecodes['Float']:
            msg = 'op: %s, dtype: %s' % (op.__name__, dt)
            fa = a.astype(dt)
            fb = b.astype(dt)
            div, rem = op(fa, fb)
            assert_equal(div, tgtdiv, err_msg=msg)
            assert_equal(rem, tgtrem, err_msg=msg)