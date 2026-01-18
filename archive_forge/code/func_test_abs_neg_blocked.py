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
def test_abs_neg_blocked(self):
    for dt, sz in [(np.float32, 11), (np.float64, 5)]:
        for out, inp, msg in _gen_alignment_data(dtype=dt, type='unary', max_size=sz):
            tgt = [ncu.absolute(i) for i in inp]
            np.absolute(inp, out=out)
            assert_equal(out, tgt, err_msg=msg)
            assert_((out >= 0).all())
            tgt = [-1 * i for i in inp]
            np.negative(inp, out=out)
            assert_equal(out, tgt, err_msg=msg)
            for v in [np.nan, -np.inf, np.inf]:
                for i in range(inp.size):
                    d = np.arange(inp.size, dtype=dt)
                    inp[:] = -d
                    inp[i] = v
                    d[i] = -v if v == -np.inf else v
                    assert_array_equal(np.abs(inp), d, err_msg=msg)
                    np.abs(inp, out=out)
                    assert_array_equal(out, d, err_msg=msg)
                    assert_array_equal(-inp, -1 * inp, err_msg=msg)
                    d = -1 * inp
                    np.negative(inp, out=out)
                    assert_array_equal(out, d, err_msg=msg)