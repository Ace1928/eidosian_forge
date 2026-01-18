import numpy as np
from numpy.testing import assert_allclose, assert_equal  #noqa
from statsmodels.stats import weightstats
import statsmodels.stats.multivariate as smmv  # pytest cannot import test_xxx
from statsmodels.stats.multivariate import confint_mvmean_fromstats
from statsmodels.tools.testing import Holder
def test_mvmean_2indep():
    x = np.asarray([[1.0, 24.0, 23.5, 1.0], [2.0, 25.0, 24.5, 1.0], [3.0, 21.0, 20.5, 1.0], [4.0, 22.0, 20.5, 1.0], [5.0, 23.0, 22.5, 1.0], [6.0, 18.0, 16.5, 1.0], [7.0, 17.0, 16.5, 1.0], [8.0, 28.0, 27.5, 1.0], [9.0, 24.0, 23.5, 1.0], [10.0, 27.0, 25.5, 1.0], [11.0, 21.0, 20.5, 1.0], [12.0, 23.0, 22.5, 1.0], [1.0, 20.0, 19.0, 0.0], [2.0, 23.0, 22.0, 0.0], [3.0, 21.0, 20.0, 0.0], [4.0, 25.0, 24.0, 0.0], [5.0, 18.0, 17.0, 0.0], [6.0, 17.0, 16.0, 0.0], [7.0, 18.0, 17.0, 0.0], [8.0, 24.0, 23.0, 0.0], [9.0, 20.0, 19.0, 0.0], [10.0, 24.0, 22.0, 0.0], [11.0, 23.0, 22.0, 0.0], [12.0, 19.0, 18.0, 0.0]])
    y = np.asarray([[1.1, 24.1, 23.4, 1.1], [1.9, 25.2, 24.3, 1.2], [3.2, 20.9, 20.2, 1.3], [4.1, 21.8, 20.6, 0.9], [5.2, 23.0, 22.7, 0.8], [6.3, 18.1, 16.8, 0.7], [7.1, 17.2, 16.5, 1.0], [7.8, 28.3, 27.4, 1.1], [9.5, 23.9, 23.3, 1.2], [10.1, 26.8, 25.2, 1.3], [10.5, 26.7, 20.6, 0.9], [12.1, 23.0, 22.7, 0.8], [1.1, 20.1, 19.0, 0.7], [1.8, 23.2, 22.0, 0.1], [3.2, 21.3, 20.3, 0.2], [4.3, 24.9, 24.2, 0.3], [5.5, 17.9, 17.1, 0.0], [5.5, 17.8, 16.0, 0.6], [7.1, 17.7, 16.7, 0.0], [7.7, 24.0, 22.8, 0.5], [9.1, 20.1, 18.9, 0.0], [10.2, 24.2, 22.3, 0.3], [11.3, 23.3, 22.2, 0.0], [11.7, 18.8, 18.1, 0.1]])
    res = smmv.test_mvmean_2indep(x, y)
    res_stata = Holder(p_F=0.6686659171701677, df_r=43, df_m=4, F=0.594263378678938, T2=2.5428944576028973)
    assert_allclose(res.statistic, res_stata.F, rtol=1e-10)
    assert_allclose(res.pvalue, res_stata.p_F, rtol=1e-10)
    assert_allclose(res.t2, res_stata.T2, rtol=1e-10)
    assert_equal(res.df, [res_stata.df_m, res_stata.df_r])