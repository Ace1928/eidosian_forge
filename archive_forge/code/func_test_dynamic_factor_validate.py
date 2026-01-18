import numpy as np
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
from numpy.testing import assert_, assert_raises, assert_equal, assert_allclose
def test_dynamic_factor_validate():
    endog = np.log(macrodata[['cpi', 'realgdp', 'realinv']]).diff().iloc[1:]
    endog = (endog - endog.mean()) / endog.std()
    mod1 = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1, error_cov_type='diagonal')
    constraints = {'loading.f1.cpi': 0.5}
    with mod1.fix_params(constraints):
        assert_(mod1._has_fixed_params)
        assert_equal(mod1._fixed_params, constraints)
        assert_equal(mod1._fixed_params_index, [0])
        assert_equal(mod1._free_params_index, [1, 2, 3, 4, 5, 6])
    res1 = mod1.fit_constrained(constraints, disp=False)
    assert_(res1._has_fixed_params)
    assert_equal(res1._fixed_params, constraints)
    assert_equal(res1._fixed_params_index, [0])
    assert_equal(res1._free_params_index, [1, 2, 3, 4, 5, 6])
    with mod1.fix_params({'L1.f1.f1': 0.5}):
        assert_(mod1._has_fixed_params)
        assert_equal(mod1._fixed_params, {'L1.f1.f1': 0.5})
        assert_equal(mod1._fixed_params_index, [6])
        assert_equal(mod1._free_params_index, [0, 1, 2, 3, 4, 5])
    mod2 = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=2, error_cov_type='diagonal')
    with pytest.raises(ValueError):
        with mod2.fix_params({'L1.f1.f1': 0.5}):
            pass
    constraints = {'L1.f1.f1': 0.3, 'L2.f1.f1': 0.1}
    with mod2.fix_params(constraints):
        assert_(mod2._has_fixed_params)
        assert_equal(mod2._fixed_params, constraints)
        assert_equal(mod2._fixed_params_index, [6, 7])
        assert_equal(mod2._free_params_index, [0, 1, 2, 3, 4, 5])
    res2 = mod2.fit_constrained(constraints, disp=False)
    assert_(res2._has_fixed_params)
    assert_equal(res2._fixed_params, constraints)
    assert_equal(res2._fixed_params_index, [6, 7])
    assert_equal(res2._free_params_index, [0, 1, 2, 3, 4, 5])
    with mod2.fix_params(constraints):
        with mod2.fix_params({'L1.f1.f1': -0.3}):
            assert_(mod2._has_fixed_params)
            assert_equal(mod2._fixed_params, {'L1.f1.f1': -0.3, 'L2.f1.f1': 0.1})
            assert_equal(mod2._fixed_params_index, [6, 7])
            assert_equal(mod2._free_params_index, [0, 1, 2, 3, 4, 5])
    mod3 = dynamic_factor.DynamicFactor(endog, k_factors=2, factor_order=1, error_cov_type='diagonal')
    with pytest.raises(ValueError):
        with mod3.fix_params({'L1.f1.f1': 0.3}):
            pass
    constraints = dict([('L1.f1.f1', 0.3), ('L1.f2.f1', 0.1), ('L1.f1.f2', -0.05), ('L1.f2.f2', 0.1)])
    with mod3.fix_params(constraints):
        assert_(mod3._has_fixed_params)
        assert_equal(mod3._fixed_params, constraints)
        assert_equal(mod3._fixed_params_index, [9, 10, 11, 12])
        assert_equal(mod3._free_params_index, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    res3 = mod3.fit_constrained(constraints, disp=False)
    assert_(res3._has_fixed_params)
    assert_equal(res3._fixed_params, constraints)
    assert_equal(res3._fixed_params_index, [9, 10, 11, 12])
    assert_equal(res3._free_params_index, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    with mod3.fix_params(constraints):
        with mod3.fix_params({'L1.f1.f1': -0.3, 'L1.f2.f2': -0.1}):
            assert_(mod3._has_fixed_params)
            assert_equal(mod3._fixed_params, dict([('L1.f1.f1', -0.3), ('L1.f2.f1', 0.1), ('L1.f1.f2', -0.05), ('L1.f2.f2', -0.1)]))
            assert_equal(mod3._fixed_params_index, [9, 10, 11, 12])
            assert_equal(mod3._free_params_index, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    mod4 = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=2, error_cov_type='diagonal', enforce_stationarity=False)
    with mod4.fix_params({'L1.f1.f1': 0.6}):
        assert_(mod4._has_fixed_params)
        assert_equal(mod4._fixed_params, {'L1.f1.f1': 0.6})
        assert_equal(mod4._fixed_params_index, [6])
        assert_equal(mod4._free_params_index, [0, 1, 2, 3, 4, 5, 7])
    mod5 = dynamic_factor.DynamicFactor(endog, k_factors=2, factor_order=1, error_cov_type='diagonal', enforce_stationarity=False)
    with mod5.fix_params({'L1.f1.f1': 0.6}):
        assert_(mod5._has_fixed_params)
        assert_equal(mod5._fixed_params, {'L1.f1.f1': 0.6})
        assert_equal(mod5._fixed_params_index, [9])
        assert_equal(mod5._free_params_index, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12])
    constraints = {'sigma2.cpi': 0.9, 'sigma2.realinv': 3}
    with mod1.fix_params(constraints):
        assert_(mod1._has_fixed_params)
        assert_equal(mod1._fixed_params, constraints)
        assert_equal(mod1._fixed_params_index, [3, 5])
        assert_equal(mod1._free_params_index, [0, 1, 2, 4, 6])
    res1 = mod1.fit_constrained(constraints, disp=False)
    assert_(res1._has_fixed_params)
    assert_equal(res1._fixed_params, constraints)
    assert_equal(res1._fixed_params_index, [3, 5])
    assert_equal(res1._free_params_index, [0, 1, 2, 4, 6])
    mod6 = dynamic_factor.DynamicFactor(endog[['cpi', 'realgdp']], k_factors=1, factor_order=1, error_cov_type='unstructured')
    constraints = {'loading.f1.cpi': 1.0, 'loading.f1.realgdp': 1.0, 'cov.chol[1,1]': 0.5, 'cov.chol[2,1]': 0.1}
    with mod6.fix_params(constraints):
        assert_(mod6._has_fixed_params)
        assert_equal(mod6._fixed_params, constraints)
        assert_equal(mod6._fixed_params_index, [0, 1, 2, 3])
        assert_equal(mod6._free_params_index, [4, 5])
    res6 = mod6.fit_constrained(constraints, disp=False)
    assert_(res6._has_fixed_params)
    assert_equal(res6._fixed_params, constraints)
    assert_equal(res6._fixed_params_index, [0, 1, 2, 3])
    assert_equal(res6._free_params_index, [4, 5])