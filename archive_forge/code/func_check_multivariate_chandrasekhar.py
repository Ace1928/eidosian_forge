import numpy as np
import pandas as pd
from .results import results_varmax
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.kalman_filter import (
from numpy.testing import assert_allclose
import pytest
def check_multivariate_chandrasekhar(filter_univariate=False, gen_obs_cov=False, memory_conserve=False, **kwargs):
    index = pd.date_range('1960-01-01', '1982-10-01', freq='QS')
    dta = pd.DataFrame(results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'], index=index)
    dta['dln_inv'] = np.log(dta['inv']).diff()
    dta['dln_inc'] = np.log(dta['inc']).diff()
    dta['dln_consump'] = np.log(dta['consump']).diff()
    endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc']]
    mod_orig = varmax.VARMAX(endog, **kwargs)
    mod_chand = varmax.VARMAX(endog, **kwargs)
    mod_chand.ssm.filter_chandrasekhar = True
    params = mod_orig.start_params
    mod_orig.ssm.filter_univariate = filter_univariate
    mod_chand.ssm.filter_univariate = filter_univariate
    if gen_obs_cov:
        mod_orig['obs_cov'] = np.array([[1.0, 0.5], [0.5, 1.0]])
        mod_chand['obs_cov'] = np.array([[1.0, 0.5], [0.5, 1.0]])
    if memory_conserve:
        mod_orig.ssm.set_conserve_memory(MEMORY_CONSERVE & ~MEMORY_NO_LIKELIHOOD)
        mod_chand.ssm.set_conserve_memory(MEMORY_CONSERVE & ~MEMORY_NO_LIKELIHOOD)
        res_chand = mod_chand.filter(params)
        res_orig = mod_orig.filter(params)
    else:
        res_chand = mod_chand.smooth(params)
        res_orig = mod_orig.smooth(params)
    check_output(res_chand, res_orig, memory_conserve=memory_conserve)