import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat
from .results.el_results import DescStatRes
def test_ci_kurt(self):
    kurt_ci = self.res1.ci_kurt(upper_bound=0.5, lower_bound=-1.5)
    lower_lim = kurt_ci[0]
    upper_lim = kurt_ci[1]
    ul_pval = self.res1.test_kurt(upper_lim)[1]
    ll_pval = self.res1.test_kurt(lower_lim)[1]
    assert_almost_equal(ul_pval, 0.05, 4)
    assert_almost_equal(ll_pval, 0.05, 4)