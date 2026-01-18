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
def test_basis_element_quadratic(self):
    xx = np.linspace(-1, 4, 20)
    b = BSpline.basis_element(t=[0, 1, 2, 3])
    assert_allclose(b(xx), splev(xx, (b.t, b.c, b.k)), atol=1e-14)
    assert_allclose(b(xx), B_0123(xx), atol=1e-14)
    b = BSpline.basis_element(t=[0, 1, 1, 2])
    xx = np.linspace(0, 2, 10)
    assert_allclose(b(xx), np.where(xx < 1, xx * xx, (2.0 - xx) ** 2), atol=1e-14)