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
def test_2x3(self):
    u = [1, 2]
    v = [3, 4, 5]
    z = np.array([10, -5, -2])
    cp = np.cross(u, v)
    assert_equal(cp, z)
    cp = np.cross(v, u)
    assert_equal(cp, -z)