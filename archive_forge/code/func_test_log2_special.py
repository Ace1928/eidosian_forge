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
def test_log2_special(self):
    assert_equal(np.log2(1.0), 0.0)
    assert_equal(np.log2(np.inf), np.inf)
    assert_(np.isnan(np.log2(np.nan)))
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', RuntimeWarning)
        assert_(np.isnan(np.log2(-1.0)))
        assert_(np.isnan(np.log2(-np.inf)))
        assert_equal(np.log2(0.0), -np.inf)
        assert_(w[0].category is RuntimeWarning)
        assert_(w[1].category is RuntimeWarning)
        assert_(w[2].category is RuntimeWarning)