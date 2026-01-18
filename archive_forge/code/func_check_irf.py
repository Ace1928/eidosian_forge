import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from statsmodels.tsa.statespace import varmax
from .results import results_var_R
def check_irf(test, mod, results, params=None):
    Sigma_u_mle = mod['state_cov']
    nobs_effective = mod.nobs - mod.k_ar
    df_resid = nobs_effective - (mod.k_ar * mod.k_endog + mod.k_trend + mod.k_exog)
    Sigma_u = Sigma_u_mle * nobs_effective / df_resid
    L = np.linalg.cholesky(Sigma_u)
    if params is None:
        params = np.copy(results['params'])
    params[-6:] = L[np.tril_indices_from(L)]
    res = mod.smooth(params)
    for i in range(3):
        impulse_to = endog.columns[i]
        columns = ['{}.irf.{}.{}'.format(test, impulse_to, name) for name in endog.columns]
        assert_allclose(res.impulse_responses(10, i), results_var_R_output[columns])
        columns = ['{}.irf.ortho.{}.{}'.format(test, impulse_to, name) for name in endog.columns]
        assert_allclose(res.impulse_responses(10, i, orthogonalized=True), results_var_R_output[columns])
        columns = ['{}.irf.cumu.{}.{}'.format(test, impulse_to, name) for name in endog.columns]
        result = res.impulse_responses(10, i, orthogonalized=True, cumulative=True)
        assert_allclose(result, results_var_R_output[columns])