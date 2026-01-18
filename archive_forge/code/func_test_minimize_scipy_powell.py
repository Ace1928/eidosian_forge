from statsmodels.compat.scipy import SP_LT_15, SP_LT_17
import pytest
from numpy.testing import assert_
from numpy.testing import assert_almost_equal
from statsmodels.base.optimizer import (
@pytest.mark.skipif(SP_LT_15, reason='Powell bounds support added in SP 1.5')
def test_minimize_scipy_powell():
    func = fit_funcs['minimize']
    xopt, _ = func(dummy_bounds_constraint_func, None, (3, 4.5), (), {'min_method': 'Powell', 'bounds': dummy_bounds_tight()}, hess=None, full_output=False, disp=0)
    assert_almost_equal(xopt, [2, 3.5], 4)