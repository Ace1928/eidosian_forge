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
def test_non_finite_scalar(self):
    assert_(np.isclose(np.inf, -np.inf) is np.False_)
    assert_(np.isclose(0, np.inf) is np.False_)
    assert_(type(np.isclose(0, np.inf)) is np.bool_)