from statsmodels.compat.scipy import SP_LT_15, SP_LT_17
import pytest
from numpy.testing import assert_
from numpy.testing import assert_almost_equal
from statsmodels.base.optimizer import (
def test_minimize_scipy_slsqp():
    func = fit_funcs['minimize']
    xopt, _ = func(dummy_bounds_constraint_func, None, (2.0, 0.0), (), {'min_method': 'SLSQP', 'bounds': dummy_bounds(), 'constraints': dummy_constraints()}, hess=None, full_output=False, disp=0)
    assert_almost_equal(xopt, [1.4, 1.7], 4)