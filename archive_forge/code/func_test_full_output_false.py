from statsmodels.compat.scipy import SP_LT_15, SP_LT_17
import pytest
from numpy.testing import assert_
from numpy.testing import assert_almost_equal
from statsmodels.base.optimizer import (
@pytest.mark.smoke
def test_full_output_false(reset_randomstate):
    for method in fit_funcs:
        func = fit_funcs[method]
        if method == 'newton':
            xopt, retvals = func(dummy_func, dummy_score, [1.0], (), {}, hess=dummy_hess, full_output=False, disp=0)
        else:
            xopt, retvals = func(dummy_func, dummy_score, [1.0], (), {}, full_output=False, disp=0)
        assert_(retvals is None)
        if method == 'powell' and SP_LT_15:
            assert_(xopt.shape == () and xopt.size == 1)
        else:
            assert_(len(xopt) == 1)