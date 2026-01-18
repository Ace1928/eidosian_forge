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
def test_degenerate_case_multidimensional(self):
    x = np.array([0, 1, 2])
    y = np.vstack((x, x ** 2)).T
    ak = Akima1DInterpolator(x, y)
    x_eval = np.array([0.5, 1.5])
    y_eval = ak(x_eval)
    assert_allclose(y_eval, np.vstack((x_eval, x_eval ** 2)).T)