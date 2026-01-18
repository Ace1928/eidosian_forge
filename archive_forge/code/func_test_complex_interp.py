import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
def test_complex_interp(self):
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5) + (1 + np.linspace(0, 1, 5)) * 1j
    x0 = 0.3
    y0 = x0 + (1 + x0) * 1j
    assert_almost_equal(np.interp(x0, x, y), y0)
    x0 = -1
    left = 2 + 3j
    assert_almost_equal(np.interp(x0, x, y, left=left), left)
    x0 = 2.0
    right = 2 + 3j
    assert_almost_equal(np.interp(x0, x, y, right=right), right)
    x = [1, 2, 2.5, 3, 4]
    xp = [1, 2, 3, 4]
    fp = [1, 2 + 1j, np.inf, 4]
    y = [1, 2 + 1j, np.inf + 0.5j, np.inf, 4]
    assert_almost_equal(np.interp(x, xp, fp), y)
    x = [-180, -170, -185, 185, -10, -5, 0, 365]
    xp = [190, -190, 350, -350]
    fp = [5 + 1j, 10 + 2j, 3 + 3j, 4 + 4j]
    y = [7.5 + 1.5j, 5.0 + 1j, 8.75 + 1.75j, 6.25 + 1.25j, 3.0 + 3j, 3.25 + 3.25j, 3.5 + 3.5j, 3.75 + 3.75j]
    assert_almost_equal(np.interp(x, xp, fp, period=360), y)