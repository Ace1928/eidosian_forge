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
def bspline(x, t, c, k):
    n = len(t) - k - 1
    assert n >= k + 1 and len(c) >= n
    return sum((c[i] * B(x, k, i, t) for i in range(n)))