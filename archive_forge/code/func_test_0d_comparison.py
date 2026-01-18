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
def test_0d_comparison(self):
    x = 3
    y = piecewise(x, [x <= 3, x > 3], [4, 0])
    assert_equal(y, 4)
    x = 4
    y = piecewise(x, [x <= 3, (x > 3) * (x <= 5), x > 5], [1, 2, 3])
    assert_array_equal(y, 2)
    assert_raises_regex(ValueError, '2 or 3 functions are expected', piecewise, x, [x <= 3, x > 3], [1])
    assert_raises_regex(ValueError, '2 or 3 functions are expected', piecewise, x, [x <= 3, x > 3], [1, 1, 1, 1])