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
@pytest.mark.parametrize('absolute_sigma', [False, True])
def test_curvefit_scalar_sigma(self, absolute_sigma):

    def func(x, a, b):
        return a * x + b
    x, y = (self.x, self.y)
    _, pcov1 = curve_fit(func, x, y, sigma=2, absolute_sigma=absolute_sigma)
    _, pcov2 = curve_fit(func, x, y, sigma=np.full_like(y, 2), absolute_sigma=absolute_sigma)
    assert np.all(pcov1 == pcov2)