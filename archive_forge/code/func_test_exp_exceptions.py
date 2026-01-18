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
@pytest.mark.xfail(_glibc_older_than('2.17'), reason='Older glibc versions may not raise appropriate FP exceptions')
def test_exp_exceptions(self):
    with np.errstate(over='raise'):
        assert_raises(FloatingPointError, np.exp, np.float16(11.0899))
        assert_raises(FloatingPointError, np.exp, np.float32(100.0))
        assert_raises(FloatingPointError, np.exp, np.float32(1e+19))
        assert_raises(FloatingPointError, np.exp, np.float64(800.0))
        assert_raises(FloatingPointError, np.exp, np.float64(1e+19))
    with np.errstate(under='raise'):
        assert_raises(FloatingPointError, np.exp, np.float16(-17.5))
        assert_raises(FloatingPointError, np.exp, np.float32(-1000.0))
        assert_raises(FloatingPointError, np.exp, np.float32(-1e+19))
        assert_raises(FloatingPointError, np.exp, np.float64(-1000.0))
        assert_raises(FloatingPointError, np.exp, np.float64(-1e+19))