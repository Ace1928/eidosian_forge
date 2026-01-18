import numpy as np
import pandas as pd
from statsmodels.tools.tools import Bunch
from .results import results_varmax
from statsmodels.tsa.statespace import sarimax, varmax
from numpy.testing import assert_raises, assert_allclose
def test_concentrated_loglike_sarimax():
    nobs = 30
    np.random.seed(28953)
    endog = np.random.normal(size=nobs)
    kwargs = {}
    out = get_sarimax_models(endog)
    assert_allclose(out.res_conc.llf, out.res_orig.llf)
    assert_allclose(out.res_conc.llf_obs, out.res_orig.llf_obs)
    assert_allclose(out.mod_conc.loglike(out.params_conc), out.mod_orig.loglike(out.params_orig))
    assert_allclose(out.mod_conc.loglikeobs(out.params_conc), out.mod_orig.loglikeobs(out.params_orig))
    endog[2:10] = np.nan
    out = get_sarimax_models(endog)
    assert_allclose(out.res_conc.llf, out.res_orig.llf)
    assert_allclose(out.res_conc.llf_obs, out.res_orig.llf_obs)
    assert_allclose(out.mod_conc.loglike(out.params_conc), out.mod_orig.loglike(out.params_orig))
    assert_allclose(out.mod_conc.loglikeobs(out.params_conc), out.mod_orig.loglikeobs(out.params_orig))
    kwargs['seasonal_order'] = (1, 1, 1, 2)
    out = get_sarimax_models(endog, **kwargs)
    assert_allclose(out.res_conc.llf, out.res_orig.llf)
    assert_allclose(out.res_conc.llf_obs[2:], out.res_orig.llf_obs[2:])
    assert_allclose(out.mod_conc.loglike(out.params_conc), out.mod_orig.loglike(out.params_orig))
    assert_allclose(out.mod_conc.loglikeobs(out.params_conc)[2:], out.mod_orig.loglikeobs(out.params_orig)[2:])
    kwargs['loglikelihood_burn'] = 5
    kwargs['trend'] = 'c'
    kwargs['exog'] = np.arange(nobs)
    out = get_sarimax_models(endog, **kwargs)
    assert_allclose(out.res_conc.llf, out.res_orig.llf)
    assert_allclose(out.res_conc.llf_obs[2:], out.res_orig.llf_obs[2:])
    assert_allclose(out.mod_conc.loglike(out.params_conc), out.mod_orig.loglike(out.params_orig))
    assert_allclose(out.mod_conc.loglikeobs(out.params_conc)[2:], out.mod_orig.loglikeobs(out.params_orig)[2:])