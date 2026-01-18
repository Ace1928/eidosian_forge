import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import cancer
from statsmodels.emplike.originregress import ELOriginRegress
from .results.el_results import OriginResults
def test_ci_beta(self):
    ci = self.res1.conf_int_el(1)
    ll = ci[0]
    ul = ci[1]
    llf_low = np.sum(np.log(self.res1.el_test([ll], [1], return_weights=1)[2]))
    llf_high = np.sum(np.log(self.res1.el_test([ul], [1], return_weights=1)[2]))
    assert_almost_equal(llf_low, self.res2.test_llf_conf, 4)
    assert_almost_equal(llf_high, self.res2.test_llf_conf, 4)