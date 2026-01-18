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
def test_n(self):
    x = list(range(3))
    assert_raises(ValueError, diff, x, n=-1)
    output = [diff(x, n=n) for n in range(1, 5)]
    expected = [[1, 1], [0], [], []]
    assert_(diff(x, n=0) is x)
    for n, (expected, out) in enumerate(zip(expected, output), start=1):
        assert_(type(out) is np.ndarray)
        assert_array_equal(out, expected)
        assert_equal(out.dtype, np.int_)
        assert_equal(len(out), max(0, len(x) - n))