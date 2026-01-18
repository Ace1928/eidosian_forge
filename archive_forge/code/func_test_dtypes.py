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
def test_dtypes(self):
    x = np.arange(-3, 5)
    y = 1.5 * x + 3.0 + 0.5 * np.sin(x)

    def func(x, a, b):
        return a * x + b
    for method in ['lm', 'trf', 'dogbox']:
        for dtx in [np.float32, np.float64]:
            for dty in [np.float32, np.float64]:
                x = x.astype(dtx)
                y = y.astype(dty)
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                p, cov = curve_fit(func, x, y, method=method)
                assert np.isfinite(cov).all()
                assert not np.allclose(p, 1)