import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.stats.sandwich_covariance as sw
def test_hac_simple():
    from statsmodels.datasets import macrodata
    d2 = macrodata.load_pandas().data
    g_gdp = 400 * np.diff(np.log(d2['realgdp'].values))
    g_inv = 400 * np.diff(np.log(d2['realinv'].values))
    exogg = add_constant(np.c_[g_gdp, d2['realint'][:-1].values])
    res_olsg = OLS(g_inv, exogg).fit()
    cov1_r = [[+1.406438998786788, -0.31803287070833297, -0.06062111121648861], [-0.3180328707083329, 0.10973083489998187, +0.000395311760301478], [-0.06062111121648865, 0.0003953117603014895, +0.08751152891247099]]
    cov2_r = [[+1.3855512908840137, -0.3133096102522685, -0.05972079768357048], [-0.3133096102522685, +0.10810116903513062, +0.000389440793564339], [-0.0597207976835705, +0.000389440793564336, +0.08621185274050362]]
    cov1 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=True)
    se1 = sw.se_cov(cov1)
    cov2 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=False)
    se2 = sw.se_cov(cov2)
    assert_allclose(cov1, cov1_r)
    assert_allclose(cov2, cov2_r)
    assert_allclose(np.sqrt(np.diag(cov1_r)), se1)
    assert_allclose(np.sqrt(np.diag(cov2_r)), se2)
    cov3 = sw.cov_hac_simple(res_olsg, use_correction=False)
    cov4 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=False)
    assert_allclose(cov3, cov4)