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
def test_cumproduct(self):
    A = [[1, 2, 3], [4, 5, 6]]
    with assert_warns(DeprecationWarning):
        expected = np.array([1, 2, 6, 24, 120, 720])
        assert_(np.all(np.cumproduct(A) == expected))