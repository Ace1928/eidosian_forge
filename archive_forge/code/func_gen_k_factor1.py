from statsmodels.compat.pandas import QUARTER_END
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tsa.statespace import (
def gen_k_factor1(nobs=10000, k=1, idiosyncratic_ar1=False, idiosyncratic_var=0.4, k_ar=6):
    ix = pd.period_range(start='1950-01', periods=1, freq='M')
    faux = pd.Series([0], index=ix)
    mod = sarimax.SARIMAX(faux, order=(k_ar, 0, 0))
    params = np.r_[[0] * (k_ar - 1), [0.5], 1.0]
    factor = mod.simulate(params, nobs)
    if idiosyncratic_ar1:
        mod_idio = sarimax.SARIMAX(faux, order=(1, 0, 0))
        endog = pd.concat([factor + mod_idio.simulate([0.7, idiosyncratic_var], nobs) for i in range(2 * k)], axis=1)
    else:
        endog = pd.concat([factor + np.random.normal(scale=idiosyncratic_var ** 0.5, size=nobs) for i in range(2 * k)], axis=1)
    levels_M = 1 + endog / 100
    levels_M.iloc[0] = 100
    levels_M = levels_M.cumprod()
    log_levels_M = np.log(levels_M) * 100
    log_levels_Q = np.log(levels_M)
    log_levels_Q.index = log_levels_Q.index.to_timestamp()
    log_levels_Q = log_levels_Q.resample(QUARTER_END).sum().iloc[:-1] * 100
    log_levels_Q.index = log_levels_Q.index.to_period()
    endog_M = log_levels_M.diff().iloc[1:, :k]
    if k > 1:
        endog_M.columns = ['yM%d_f1' % (i + 1) for i in range(k)]
    else:
        endog_M.columns = ['yM_f1']
    endog_Q = log_levels_Q.diff().iloc[1:, k:]
    if k > 1:
        endog_Q.columns = ['yQ%d_f1' % (i + 1) for i in range(k)]
    else:
        endog_Q.columns = ['yQ_f1']
    return (endog_M, endog_Q, factor)