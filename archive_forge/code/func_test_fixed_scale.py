import numpy as np
from numpy.testing import (
import pytest
from scipy import stats
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS, WLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.sm_exceptions import InvalidTestWarning
from statsmodels.tools.tools import add_constant
from .results import (
def test_fixed_scale(self):
    cov_type = 'fixed_scale'
    kwds = {}
    res1 = self.res_ols.get_robustcov_results(cov_type, **kwds)
    res2 = self.res_wls.get_robustcov_results(cov_type, **kwds)
    assert_allclose(res1.params, res2.params, rtol=1e-13)
    assert_allclose(res1.cov_params(), res2.cov_params(), rtol=1e-13)
    assert_allclose(res1.bse, res2.bse, rtol=1e-13)
    assert_allclose(res1.pvalues, res2.pvalues, rtol=1e-12)
    tt = res2.t_test(np.eye(len(res2.params)), cov_p=res2.normalized_cov_params)
    assert_allclose(res2.cov_params(), res2.normalized_cov_params, rtol=1e-13)
    assert_allclose(res2.bse, tt.sd, rtol=1e-13)
    assert_allclose(res2.pvalues, tt.pvalue, rtol=1e-13)
    assert_allclose(res2.tvalues, tt.tvalue, rtol=1e-13)
    mod = self.res_wls.model
    mod3 = WLS(mod.endog, mod.exog, weights=mod.weights)
    res3 = mod3.fit(cov_type=cov_type, cov_kwds=kwds)
    tt = res3.t_test(np.eye(len(res3.params)), cov_p=res3.normalized_cov_params)
    assert_allclose(res3.cov_params(), res3.normalized_cov_params, rtol=1e-13)
    assert_allclose(res3.bse, tt.sd, rtol=1e-13)
    assert_allclose(res3.pvalues, tt.pvalue, rtol=1e-13)
    assert_allclose(res3.tvalues, tt.tvalue, rtol=1e-13)