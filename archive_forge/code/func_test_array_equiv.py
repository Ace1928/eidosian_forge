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
def test_array_equiv(self):
    res = np.array_equiv(np.array([1, 2]), np.array([1, 2]))
    assert_(res)
    assert_(type(res) is bool)
    res = np.array_equiv(np.array([1, 2]), np.array([1, 2, 3]))
    assert_(not res)
    assert_(type(res) is bool)
    res = np.array_equiv(np.array([1, 2]), np.array([3, 4]))
    assert_(not res)
    assert_(type(res) is bool)
    res = np.array_equiv(np.array([1, 2]), np.array([1, 3]))
    assert_(not res)
    assert_(type(res) is bool)
    res = np.array_equiv(np.array([1, 1]), np.array([1]))
    assert_(res)
    assert_(type(res) is bool)
    res = np.array_equiv(np.array([1, 1]), np.array([[1], [1]]))
    assert_(res)
    assert_(type(res) is bool)
    res = np.array_equiv(np.array([1, 2]), np.array([2]))
    assert_(not res)
    assert_(type(res) is bool)
    res = np.array_equiv(np.array([1, 2]), np.array([[1], [2]]))
    assert_(not res)
    assert_(type(res) is bool)
    res = np.array_equiv(np.array([1, 2]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert_(not res)
    assert_(type(res) is bool)