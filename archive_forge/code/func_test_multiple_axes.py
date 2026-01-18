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
def test_multiple_axes(self):
    a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    assert_equal(np.flip(a, axis=()), a)
    b = np.array([[[5, 4], [7, 6]], [[1, 0], [3, 2]]])
    assert_equal(np.flip(a, axis=(0, 2)), b)
    c = np.array([[[3, 2], [1, 0]], [[7, 6], [5, 4]]])
    assert_equal(np.flip(a, axis=(1, 2)), c)