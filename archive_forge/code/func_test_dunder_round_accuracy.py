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
def test_dunder_round_accuracy(self):
    f = np.float64(5.1 * 10 ** 73)
    assert_(isinstance(round(f, -73), np.float64))
    assert_array_max_ulp(round(f, -73), 5.0 * 10 ** 73)
    assert_(isinstance(round(f, ndigits=-73), np.float64))
    assert_array_max_ulp(round(f, ndigits=-73), 5.0 * 10 ** 73)
    i = np.int64(501)
    assert_(isinstance(round(i, -2), np.int64))
    assert_array_max_ulp(round(i, -2), 500)
    assert_(isinstance(round(i, ndigits=-2), np.int64))
    assert_array_max_ulp(round(i, ndigits=-2), 500)