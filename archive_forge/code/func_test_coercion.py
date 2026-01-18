import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def test_coercion(self):

    def res_type(a, b):
        return np.add(a, b).dtype
    self.check_promotion_cases(res_type)
    for a in [np.array([True, False]), np.array([-3, 12], dtype=np.int8)]:
        b = 1.234 * a
        assert_equal(b.dtype, np.dtype('f8'), 'array type %s' % a.dtype)
        b = np.longdouble(1.234) * a
        assert_equal(b.dtype, np.dtype(np.longdouble), 'array type %s' % a.dtype)
        b = np.float64(1.234) * a
        assert_equal(b.dtype, np.dtype('f8'), 'array type %s' % a.dtype)
        b = np.float32(1.234) * a
        assert_equal(b.dtype, np.dtype('f4'), 'array type %s' % a.dtype)
        b = np.float16(1.234) * a
        assert_equal(b.dtype, np.dtype('f2'), 'array type %s' % a.dtype)
        b = 1.234j * a
        assert_equal(b.dtype, np.dtype('c16'), 'array type %s' % a.dtype)
        b = np.clongdouble(1.234j) * a
        assert_equal(b.dtype, np.dtype(np.clongdouble), 'array type %s' % a.dtype)
        b = np.complex128(1.234j) * a
        assert_equal(b.dtype, np.dtype('c16'), 'array type %s' % a.dtype)
        b = np.complex64(1.234j) * a
        assert_equal(b.dtype, np.dtype('c8'), 'array type %s' % a.dtype)