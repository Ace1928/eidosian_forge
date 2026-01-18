import numpy as np
from numpy import array
from numpy.testing import (assert_allclose, assert_array_equal,
import pytest
from pytest import raises
import scipy.signal._bsplines as bsp
from scipy import signal
def test_cspline1d(self):
    np.random.seed(12462)
    assert_array_equal(bsp.cspline1d(array([0])), [0.0])
    c1d = array([1.21037185, 1.86293902, 2.98834059, 4.11660378, 4.78893826])
    assert_allclose(bsp.cspline1d(array([1.0, 2, 3, 4, 5]), 1), c1d)
    c1d0 = array([0.78683946, 2.05333735, 2.99981113, 3.94741812, 5.21051638])
    assert_allclose(bsp.cspline1d(array([1.0, 2, 3, 4, 5])), c1d0)