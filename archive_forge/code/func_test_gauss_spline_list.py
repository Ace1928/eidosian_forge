import numpy as np
from numpy import array
from numpy.testing import (assert_allclose, assert_array_equal,
import pytest
from pytest import raises
import scipy.signal._bsplines as bsp
from scipy import signal
def test_gauss_spline_list(self):
    knots = [-1.0, 0.0, -1.0]
    assert_almost_equal(bsp.gauss_spline(knots, 3), array([0.15418033, 0.6909883, 0.15418033]))