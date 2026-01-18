from statsmodels.compat.pandas import QUARTER_END
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tsa.statespace import (
def gen_k_factor2(nobs=10000, k=2, idiosyncratic_ar1=False, idiosyncratic_var=0.4, k_ar=6):
    ix = pd.period_range(start='1950-01', periods=1, freq='M')
    faux = pd.DataFrame([[0, 0]], index=ix, columns=['f1', 'f2'])
    mod = varmax.VARMAX(faux, order=(k_ar, 0), trend='n')
    A = np.zeros((2, 2 * k_ar))
    A[:, -2:] = np.array([[0.5, -0.2], [0.1, 0.3]])
    Q = np.array([[1.5, 0.2], [0.2, 0.5]])
    L = np.linalg.cholesky(Q)
    params = np.r_[A.ravel(), L[np.tril_indices_from(L)]]
    factors = mod.simulate(params, nobs)
    faux = pd.Series([0], index=ix)
    mod_idio = sarimax.SARIMAX(faux, order=(1, 0, 0))
    phi = [0.7, -0.2] if idiosyncratic_ar1 else [0, 0.0]
    tmp = factors.iloc[:, 0] + factors.iloc[:, 1]
    endog_M = pd.concat([tmp.copy() for i in range(k)], axis=1)
    columns = []
    for i in range(k):
        endog_M.iloc[:, i] = endog_M.iloc[:, i] + mod_idio.simulate([phi[0], idiosyncratic_var], nobs)
        columns += [f'yM{i + 1}_f2']
    endog_M.columns = columns
    endog_Q_M = pd.concat([tmp.copy() for i in range(k)], axis=1)
    columns = []
    for i in range(k):
        endog_Q_M.iloc[:, i] = endog_Q_M.iloc[:, i] + mod_idio.simulate([phi[0], idiosyncratic_var], nobs)
        columns += [f'yQ{i + 1}_f2']
    endog_Q_M.columns = columns
    levels_M = 1 + endog_Q_M / 100
    levels_M.iloc[0] = 100
    levels_M = levels_M.cumprod()
    log_levels_Q = np.log(levels_M)
    log_levels_Q.index = log_levels_Q.index.to_timestamp()
    log_levels_Q = log_levels_Q.resample(QUARTER_END).sum().iloc[:-1] * 100
    log_levels_Q.index = log_levels_Q.index.to_period()
    endog_Q = log_levels_Q.diff()
    return (endog_M, endog_Q, factors)