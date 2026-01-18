import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
def test_checkfinite(self):
    x = np.arange(12).astype(float)
    y = x ** 2
    t = _augknt(x, 3)
    for z in [np.nan, np.inf, -np.inf]:
        y[-1] = z
        assert_raises(ValueError, make_lsq_spline, x, y, t)