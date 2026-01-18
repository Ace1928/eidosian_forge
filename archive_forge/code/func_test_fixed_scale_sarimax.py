import numpy as np
import pandas as pd
from statsmodels.tools.tools import Bunch
from .results import results_varmax
from statsmodels.tsa.statespace import sarimax, varmax
from numpy.testing import assert_raises, assert_allclose
def test_fixed_scale_sarimax():
    nobs = 30
    np.random.seed(28953)
    endog = np.random.normal(size=nobs)
    kwargs = {'seasonal_order': (1, 1, 1, 2), 'trend': 'ct', 'exog': np.sin(np.arange(nobs))}
    mod_conc = sarimax.SARIMAX(endog, concentrate_scale=True, **kwargs)
    mod_orig = sarimax.SARIMAX(endog, **kwargs)
    params = mod_orig.start_params
    params[-1] *= 1.2
    assert_raises(AssertionError, assert_allclose, mod_conc.loglike(params[:-1]), mod_orig.loglike(params))
    with mod_conc.ssm.fixed_scale(params[-1]):
        res1 = mod_conc.smooth(params[:-1])
        llf1 = mod_conc.loglike(params[:-1])
        llf_obs1 = mod_conc.loglikeobs(params[:-1])
    res2 = mod_orig.smooth(params)
    llf2 = mod_orig.loglike(params)
    llf_obs2 = mod_orig.loglikeobs(params)
    assert_allclose(res1.llf, res2.llf)
    assert_allclose(res1.llf_obs[2:], res2.llf_obs[2:])
    assert_allclose(llf1, llf2)
    assert_allclose(llf_obs1, llf_obs2)