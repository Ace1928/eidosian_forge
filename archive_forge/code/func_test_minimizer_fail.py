import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_minimizer_fail(self):
    i = 1
    self.kwargs['options'] = dict(maxiter=0)
    self.niter = 10
    res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=self.niter, disp=self.disp)
    assert_equal(res.nit + 1, res.minimization_failures)