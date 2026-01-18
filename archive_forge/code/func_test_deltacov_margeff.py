import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.regression.linear_model import OLS
from statsmodels.stats._delta_method import NonlinearDeltaCov
def test_deltacov_margeff():
    import statsmodels.discrete.tests.test_discrete as dt
    tc = dt.TestPoissonNewton()
    tc.setup_class()
    res_poi = tc.res1
    res_poi.model._derivative_exog

    def f(p):
        ex = res_poi.model.exog.mean(0)[None, :]
        fv = res_poi.model._derivative_exog(p, ex)
        return np.squeeze(fv)
    nlp = NonlinearDeltaCov(f, res_poi.params, res_poi.cov_params())
    marg = res_poi.get_margeff(at='mean')
    assert_allclose(nlp.se_vectorized()[:-1], marg.margeff_se, rtol=1e-13)
    assert_allclose(nlp.predicted()[:-1], marg.margeff, rtol=1e-13)
    nlpm = res_poi._get_wald_nonlinear(f)
    assert_allclose(nlpm.se_vectorized()[:-1], marg.margeff_se, rtol=1e-13)
    assert_allclose(nlpm.predicted()[:-1], marg.margeff, rtol=1e-13)