import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.stats.sandwich_covariance as sw
def test_cov_cluster_2groups():
    import os
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    fpath = os.path.join(cur_dir, 'test_data.txt')
    pet = np.genfromtxt(fpath)
    endog = pet[:, -1]
    group = pet[:, 0].astype(int)
    time = pet[:, 1].astype(int)
    exog = add_constant(pet[:, 2])
    res = OLS(endog, exog).fit()
    cov01, covg, covt = sw.cov_cluster_2groups(res, group, group2=time)
    bse_petw = [0.0284, 0.0284]
    bse_pet0 = [0.067, 0.0506]
    bse_pet1 = [0.0234, 0.0334]
    bse_pet01 = [0.0651, 0.0536]
    bse_0 = sw.se_cov(covg)
    bse_1 = sw.se_cov(covt)
    bse_01 = sw.se_cov(cov01)
    assert_almost_equal(bse_petw, res.HC0_se, decimal=4)
    assert_almost_equal(bse_0, bse_pet0, decimal=4)
    assert_almost_equal(bse_1, bse_pet1, decimal=4)
    assert_almost_equal(bse_01, bse_pet01, decimal=4)