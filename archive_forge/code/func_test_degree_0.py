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
def test_degree_0(self):
    xx = np.linspace(0, 1, 10)
    b = BSpline(t=[0, 1], c=[3.0], k=0)
    assert_allclose(b(xx), 3)
    b = BSpline(t=[0, 0.35, 1], c=[3, 4], k=0)
    assert_allclose(b(xx), np.where(xx < 0.35, 3, 4))