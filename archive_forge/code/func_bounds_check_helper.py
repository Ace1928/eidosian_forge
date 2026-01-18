from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
def bounds_check_helper(self, interpolant, test_array, fail_value):
    assert_raises(ValueError, interpolant, test_array)
    try:
        interpolant(test_array)
    except ValueError as err:
        assert f'{fail_value}' in str(err)