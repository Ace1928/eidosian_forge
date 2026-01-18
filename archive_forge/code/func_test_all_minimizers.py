import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_all_minimizers(self):
    i = 1
    methods = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP']
    minimizer_kwargs = copy.copy(self.kwargs)
    for method in methods:
        minimizer_kwargs['method'] = method
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=minimizer_kwargs, niter=self.niter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)