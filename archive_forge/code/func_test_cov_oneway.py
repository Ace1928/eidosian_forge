import numpy as np
from numpy.testing import assert_allclose, assert_equal  #noqa
from statsmodels.stats import weightstats
import statsmodels.stats.multivariate as smmv  # pytest cannot import test_xxx
from statsmodels.stats.multivariate import confint_mvmean_fromstats
from statsmodels.tools.testing import Holder
def test_cov_oneway():
    p_chi2 = 0.1944866419800838
    chi2 = 13.55075120374669
    df = 10
    p_F_Box = 0.1949865290585139
    df_r_Box = 18377.68924302788
    df_m_Box = 10
    F_Box = 1.354282822767436
    nobs = [32, 32]
    cov_m = np.array([[5.192540322580645, 4.545362903225806, 6.522177419354839, 5.25], [4.545362903225806, 13.184475806451612, 6.76008064516129, 6.266129032258064], [6.522177419354839, 6.76008064516129, 28.673387096774192, 14.46774193548387], [5.25, 6.266129032258064, 14.46774193548387, 16.64516129032258]])
    cov_f = np.array([[9.13608870967742, 7.549395161290322, 4.86391129032258, 4.151209677419355], [7.549395161290322, 18.60383064516129, 10.224798387096774, 5.445564516129032], [4.86391129032258, 10.224798387096774, 30.039314516129032, 13.493951612903226], [4.151209677419355, 5.445564516129032, 13.493951612903226, 27.995967741935484]])
    res = smmv.test_cov_oneway([cov_m, cov_f], nobs)
    stat, pv = res
    assert_allclose(stat, F_Box, rtol=1e-10)
    assert_allclose(pv, p_F_Box, rtol=1e-06)
    assert_allclose(res.statistic_f, F_Box, rtol=1e-10)
    assert_allclose(res.pvalue_f, p_F_Box, rtol=1e-06)
    assert_allclose(res.df_f, (df_m_Box, df_r_Box), rtol=1e-13)
    assert_allclose(res.statistic_chi2, chi2, rtol=1e-10)
    assert_allclose(res.pvalue_chi2, p_chi2, rtol=1e-06)
    assert_equal(res.df_chi2, df)