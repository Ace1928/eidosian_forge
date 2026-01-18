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
def bspline2(xy, t, c, k):
    """A naive 2D tensort product spline evaluation."""
    x, y = xy
    tx, ty = t
    nx = len(tx) - k - 1
    assert nx >= k + 1
    ny = len(ty) - k - 1
    assert ny >= k + 1
    return sum((c[ix, iy] * B(x, k, ix, tx) * B(y, k, iy, ty) for ix in range(nx) for iy in range(ny)))