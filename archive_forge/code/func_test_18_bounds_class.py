import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
def test_18_bounds_class(self):

    def f(x):
        return numpy.square(x).sum()
    lb = [-6.0, 1.0, -5.0]
    ub = [-1.0, 3.0, 5.0]
    bounds_old = list(zip(lb, ub))
    bounds_new = Bounds(lb, ub)
    res_old_bounds = shgo(f, bounds_old)
    res_new_bounds = shgo(f, bounds_new)
    assert res_new_bounds.nfev == res_old_bounds.nfev
    assert res_new_bounds.message == res_old_bounds.message
    assert res_new_bounds.success == res_old_bounds.success
    x_opt = numpy.array([-1.0, 1.0, 0.0])
    numpy.testing.assert_allclose(res_new_bounds.x, x_opt)
    numpy.testing.assert_allclose(res_new_bounds.x, res_old_bounds.x)