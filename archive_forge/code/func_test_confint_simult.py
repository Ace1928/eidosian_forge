import numpy as np
from numpy.testing import assert_allclose, assert_equal  #noqa
from statsmodels.stats import weightstats
import statsmodels.stats.multivariate as smmv  # pytest cannot import test_xxx
from statsmodels.stats.multivariate import confint_mvmean_fromstats
from statsmodels.tools.testing import Holder
def test_confint_simult():
    m = [526.29, 54.69, 25.13]
    cov = [[5808.06, 597.84, 222.03], [597.84, 126.05, 23.39], [222.03, 23.39, 23.11]]
    nobs = 87
    res_ci = confint_mvmean_fromstats(m, cov, nobs, lin_transf=np.eye(3), simult=True)
    cii = [confint_mvmean_fromstats(m, cov, nobs, lin_transf=np.eye(3)[i], simult=True)[:2] for i in range(3)]
    cii = np.array(cii).squeeze()
    res_ci_book = np.array([[503.06, 550.12], [51.22, 58.16], [23.65, 26.61]])
    assert_allclose(res_ci[0], res_ci_book[:, 0], rtol=0.001)
    assert_allclose(res_ci[0], res_ci_book[:, 0], rtol=0.001)
    assert_allclose(res_ci[0], cii[:, 0], rtol=1e-13)
    assert_allclose(res_ci[1], cii[:, 1], rtol=1e-13)
    res_constr = confint_mvmean_fromstats(m, cov, nobs, lin_transf=[0, 1, -1], simult=True)
    assert_allclose(res_constr[0], 29.56 - 3.12, rtol=0.001)
    assert_allclose(res_constr[1], 29.56 + 3.12, rtol=0.001)
    lt = [[0, 1, -1], [0, -1, 1], [0, 2, -2]]
    res_constr2 = confint_mvmean_fromstats(m, cov, nobs, lin_transf=lt, simult=True)
    lows = (res_constr[0], -res_constr[1], 2 * res_constr[0])
    upps = (res_constr[1], -res_constr[0], 2 * res_constr[1])
    lows = np.asarray(lows).squeeze()
    upps = np.asarray(upps).squeeze()
    assert_allclose(res_constr2[0], lows, rtol=1e-13)
    assert_allclose(res_constr2[1], upps, rtol=1e-13)