import numpy as np
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.stats._diagnostic_other import CMTNewey, CMTTauchen
import statsmodels.stats._diagnostic_other as diao
@classmethod
def attach_moment_conditions(cls):
    res_ols = cls.res_ols
    x = cls.exog_full
    x1 = cls.exog_add
    nobs, k_constraints = x1.shape
    moms_obs = res_ols.resid[:, None] * x
    moms = moms_obs.sum(0)
    cov_moms = res_ols.mse_resid * x.T.dot(x)
    cov_moms *= res_ols.df_resid / nobs
    weights = np.linalg.inv(cov_moms)
    weights[:, -k_constraints:] = 0
    weights[-k_constraints:, :] = 0
    k_moms = moms.shape[0]
    L = np.eye(k_moms)[-k_constraints:]
    moms_deriv = cov_moms[:, :-k_constraints]
    covm = moms_obs.T.dot(moms_obs)
    cls.nobs = nobs
    cls.moms = moms
    cls.moms_obs = moms_obs
    cls.cov_moms = cov_moms
    cls.covm = covm
    cls.moms_deriv = moms_deriv
    cls.weights = weights
    cls.L = L