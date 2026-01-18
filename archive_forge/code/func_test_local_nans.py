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
def test_local_nans(self):
    x = np.arange(10).astype(float)
    y = x.copy()
    y[6] = np.nan
    for kind in ('zero', 'slinear'):
        ir = interp1d(x, y, kind=kind)
        vals = ir([4.9, 7.0])
        assert_(np.isfinite(vals).all())