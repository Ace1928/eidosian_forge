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
def test_monotonic(self):
    x = [-1, 0, 1, 2]
    bins = [0, 0, 1]
    assert_array_equal(digitize(x, bins, False), [0, 2, 3, 3])
    assert_array_equal(digitize(x, bins, True), [0, 0, 2, 3])
    bins = [1, 1, 0]
    assert_array_equal(digitize(x, bins, False), [3, 2, 0, 0])
    assert_array_equal(digitize(x, bins, True), [3, 3, 2, 0])
    bins = [1, 1, 1, 1]
    assert_array_equal(digitize(x, bins, False), [0, 0, 4, 4])
    assert_array_equal(digitize(x, bins, True), [0, 0, 0, 4])
    bins = [0, 0, 1, 0]
    assert_raises(ValueError, digitize, x, bins)
    bins = [1, 1, 0, 1]
    assert_raises(ValueError, digitize, x, bins)