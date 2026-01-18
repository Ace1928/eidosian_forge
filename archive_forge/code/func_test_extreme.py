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
def test_extreme(self):
    x = [[1e-100, 1e+100], [1e+100, 1e-100]]
    with np.errstate(all='raise'):
        c = corrcoef(x)
    assert_array_almost_equal(c, np.array([[1.0, -1.0], [-1.0, 1.0]]))
    assert_(np.all(np.abs(c) <= 1.0))