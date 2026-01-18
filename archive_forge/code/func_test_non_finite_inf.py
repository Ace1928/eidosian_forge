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
def test_non_finite_inf(self, sc):
    """ Test that interp between opposite infs gives nan """
    assert_equal(np.interp(0.5, [-np.inf, +np.inf], sc([0, 10])), sc(np.nan))
    assert_equal(np.interp(0.5, [0, 1], sc([-np.inf, +np.inf])), sc(np.nan))
    assert_equal(np.interp(0.5, [0, 1], sc([+np.inf, -np.inf])), sc(np.nan))
    assert_equal(np.interp(0.5, [-np.inf, +np.inf], sc([10, 10])), sc(10))