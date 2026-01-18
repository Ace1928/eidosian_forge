import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
def test_maxfev_and_bounds(self):
    x = np.arange(0, 10)
    y = 2 * x
    popt1, _ = curve_fit(lambda x, p: p * x, x, y, bounds=(0, 3), maxfev=100)
    popt2, _ = curve_fit(lambda x, p: p * x, x, y, bounds=(0, 3), max_nfev=100)
    assert_allclose(popt1, 2, atol=1e-14)
    assert_allclose(popt2, 2, atol=1e-14)