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
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
def test_float_remainder_overflow(self):
    a = np.finfo(np.float64).tiny
    with np.errstate(over='ignore', invalid='ignore'):
        div, mod = np.divmod(4, a)
        np.isinf(div)
        assert_(mod == 0)
    with np.errstate(over='raise', invalid='ignore'):
        assert_raises(FloatingPointError, np.divmod, 4, a)
    with np.errstate(invalid='raise', over='ignore'):
        assert_raises(FloatingPointError, np.divmod, 4, a)