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
def test_int_xy(self):
    x = np.arange(10).astype(int)
    y = np.arange(10).astype(int)
    t = _augknt(x, k=1)
    make_lsq_spline(x, y, t, k=1)