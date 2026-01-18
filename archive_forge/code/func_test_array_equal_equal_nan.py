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
def test_array_equal_equal_nan(self):
    a1 = np.array([1, 2, np.nan])
    a2 = np.array([1, np.nan, 2])
    a3 = np.array([1, 2, np.inf])
    assert_(not np.array_equal(a1, a1))
    assert_(np.array_equal(a1, a1, equal_nan=True))
    assert_(not np.array_equal(a1, a2, equal_nan=True))
    assert_(not np.array_equal(a1, a3, equal_nan=True))
    a = np.array(np.nan)
    assert_(not np.array_equal(a, a))
    assert_(np.array_equal(a, a, equal_nan=True))
    a = np.array([1, 2, 3], dtype=int)
    assert_(np.array_equal(a, a))
    assert_(np.array_equal(a, a, equal_nan=True))
    a = np.array([[0, 1], [np.nan, 1]])
    assert_(not np.array_equal(a, a))
    assert_(np.array_equal(a, a, equal_nan=True))
    a, b = [np.array([1 + 1j])] * 2
    a.real, b.imag = (np.nan, np.nan)
    assert_(not np.array_equal(a, b, equal_nan=False))
    assert_(np.array_equal(a, b, equal_nan=True))