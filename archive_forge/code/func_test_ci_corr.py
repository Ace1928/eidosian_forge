import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat
from .results.el_results import DescStatRes
def test_ci_corr(self):
    corr_ci = self.mvres1.ci_corr()
    lower_lim = corr_ci[0]
    upper_lim = corr_ci[1]
    ul_pval = self.mvres1.test_corr(upper_lim)[1]
    ll_pval = self.mvres1.test_corr(lower_lim)[1]
    assert_almost_equal(ul_pval, 0.05, 4)
    assert_almost_equal(ll_pval, 0.05, 4)