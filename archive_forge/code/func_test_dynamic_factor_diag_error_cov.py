import numpy as np
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
from numpy.testing import assert_, assert_raises, assert_equal, assert_allclose
def test_dynamic_factor_diag_error_cov():
    endog = np.log(macrodata[['cpi', 'realgdp']]).diff().iloc[1:]
    endog = (endog - endog.mean()) / endog.std()
    mod1 = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1, error_cov_type='diagonal')
    mod2 = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1, error_cov_type='unstructured')
    constraints = {'cov.chol[2,1]': 0}
    start_params = [-4.5e-06, -1e-05, 0.99, 0.99, -0.14]
    res1 = mod1.fit(start_params=start_params, disp=False)
    res2 = mod2.fit_constrained(constraints, start_params=res1.params, includes_fixed=False, disp=False)
    assert_equal(res1.fixed_params, [])
    assert_equal(res2.fixed_params, ['cov.chol[2,1]'])
    param_vals = np.asarray(res1.params)
    params = np.r_[param_vals[:2], param_vals[2:4] ** 0.5, param_vals[4]]
    desired = np.r_[params[:3], 0, params[3:]]
    assert_allclose(res2.params, desired, atol=1e-05)
    with mod2.fix_params(constraints):
        res2 = mod2.smooth(params)
    check_results(res1, res2, check_params=False)