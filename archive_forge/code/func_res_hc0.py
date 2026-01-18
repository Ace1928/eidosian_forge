import numpy as np
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.stats._diagnostic_other import CMTNewey, CMTTauchen
import statsmodels.stats._diagnostic_other as diao
def res_hc0(self):
    res_ols = self.res_ols
    nobs = self.nobs
    moms = self.moms
    moms_obs = self.moms_obs
    cov_moms = self.cov_moms
    covm = self.covm
    moms_deriv = self.moms_deriv
    weights = self.weights
    L = self.L
    x0 = res_ols.model.exog
    x1 = self.exog_add
    res_all = []
    tres = diao.cm_test_robust(resid=res_ols.resid, resid_deriv=x0, instruments=x1, weights=1)
    res_all.append(('Wooldridge', tres[:2]))
    tres = CMTNewey(moms, covm, moms_deriv, weights, L).chisquare
    res_all.append(('Newey', tres))
    tres = CMTTauchen(moms[:-2], cov_moms[:-2, :-2], moms[-2:], cov_moms[-2:, :-2], covm).chisquare
    res_all.append(('Tauchen', tres))
    tres = diao.lm_robust_subset(moms[-2:], 2, cov_moms, covm)
    res_all.append(('score subset QMLE', tres))
    tres = diao.lm_robust(moms, np.eye(moms.shape[0])[-2:], np.linalg.inv(cov_moms), covm)
    res_all.append(('scoreB QMLE', tres))
    Ainv = np.linalg.inv(cov_moms)
    vv = Ainv.dot(covm).dot(Ainv)
    tres = diao.lm_robust(moms, np.eye(moms.shape[0])[-2:], np.linalg.inv(cov_moms), None, cov_params=vv)
    res_all.append(('scoreV QMLE', tres))
    tres = diao.conditional_moment_test_generic(moms_obs[:, -2:], cov_moms[-2:, :-2], moms_obs[:, :-2], cov_moms[:-2, :-2])
    tres_ = (tres.stat_cmt, tres.pval_cmt)
    res_all.append(('cmt', tres_))
    x = self.exog_full
    hess_unscaled = x.T.dot(x)
    tres = diao.conditional_moment_test_generic(moms_obs[:, -2:], hess_unscaled[-2:, :-2], moms_obs[:, :-2], hess_unscaled[:-2, :-2])
    tres_ = (tres.stat_cmt, tres.pval_cmt)
    res_all.append(('cmt', tres_))
    score_deriv_uu = cov_moms[:-2, :-2]
    score_deriv_cu = cov_moms[-2:, :-2]
    cov_score_cc = covm[-2:, -2:]
    cov_score_cu = covm[-2:, :-2]
    cov_score_uu = covm[:-2, :-2]
    (moms[-2:], 2, cov_moms, covm)
    tres = diao.lm_robust_subset_parts(moms[-2:], 2, score_deriv_uu, score_deriv_cu, cov_score_cc, cov_score_cu, cov_score_uu)
    res_all.append(('score subset_parts QMLE', tres))
    params_deriv = np.eye(x.shape[1], x.shape[1] - 2)
    score = moms
    score_deriv = cov_moms
    cov_score = covm
    tres = diao.lm_robust_reparameterized(score, params_deriv, score_deriv, cov_score)
    res_all.append(('score reparam QMLE', tres))
    return res_all