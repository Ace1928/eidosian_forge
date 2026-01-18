import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_float_spacing(self):
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, -10.0])
    z = np.array([100.0, 2.0, -1.0])
    w = np.array([-100.0, 2.0, 1.0])
    assert_equal(repr(x), 'array([1., 2., 3.])')
    assert_equal(repr(y), 'array([  1.,   2., -10.])')
    assert_equal(repr(np.array(y[0])), 'array(1.)')
    assert_equal(repr(np.array(y[-1])), 'array(-10.)')
    assert_equal(repr(z), 'array([100.,   2.,  -1.])')
    assert_equal(repr(w), 'array([-100.,    2.,    1.])')
    assert_equal(repr(np.array([np.nan, np.inf])), 'array([nan, inf])')
    assert_equal(repr(np.array([np.nan, -np.inf])), 'array([ nan, -inf])')
    x = np.array([np.inf, 100000, 1.1234])
    y = np.array([np.inf, 100000, -1.1234])
    z = np.array([np.inf, 1.1234, -1e+120])
    np.set_printoptions(precision=2)
    assert_equal(repr(x), 'array([     inf, 1.00e+05, 1.12e+00])')
    assert_equal(repr(y), 'array([      inf,  1.00e+05, -1.12e+00])')
    assert_equal(repr(z), 'array([       inf,  1.12e+000, -1.00e+120])')