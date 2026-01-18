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
def test_badargs(self):
    f_2d = np.arange(25).reshape(5, 5)
    x = np.cumsum(np.ones(5))
    assert_raises(ValueError, gradient, f_2d, x, np.ones(2))
    assert_raises(ValueError, gradient, f_2d, 1, np.ones(2))
    assert_raises(ValueError, gradient, f_2d, np.ones(2), np.ones(2))
    assert_raises(TypeError, gradient, f_2d, x)
    assert_raises(TypeError, gradient, f_2d, x, axis=(0, 1))
    assert_raises(TypeError, gradient, f_2d, x, x, x)
    assert_raises(TypeError, gradient, f_2d, 1, 1, 1)
    assert_raises(TypeError, gradient, f_2d, x, x, axis=1)
    assert_raises(TypeError, gradient, f_2d, 1, 1, axis=1)