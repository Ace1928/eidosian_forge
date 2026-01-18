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
def test_eval_2d(self):
    x = np.arange(0.0, 11.0)
    y = np.array([0.0, 2.0, 1.0, 3.0, 2.0, 6.0, 5.5, 5.5, 2.7, 5.1, 3.0])
    y = np.column_stack((y, 2.0 * y))
    ak = Akima1DInterpolator(x, y)
    xi = np.array([0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2, 8.6, 9.9, 10.0])
    yi = np.array([0.0, 1.375, 2.0, 1.5, 1.953125, 2.484375, 4.136363636363637, 5.980362391033624, 5.506729151646239, 5.203136745974525, 4.179655415901708, 3.411038659793813, 3.0])
    yi = np.column_stack((yi, 2.0 * yi))
    assert_allclose(ak(xi), yi)