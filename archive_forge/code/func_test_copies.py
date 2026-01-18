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
def test_copies(self):
    A = np.array([[1, 2], [3, 4]])
    Ar1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    assert_equal(np.resize(A, (2, 4)), Ar1)
    Ar2 = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    assert_equal(np.resize(A, (4, 2)), Ar2)
    Ar3 = np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]])
    assert_equal(np.resize(A, (4, 3)), Ar3)