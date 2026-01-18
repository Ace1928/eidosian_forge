import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
@pytest.mark.parametrize('key,val', sorted(rmethods.items()))
def test_multi_pvalcorrection_rmethods(self, key, val):
    res_multtest = self.res2
    pval0 = res_multtest[:, 0]
    if val[1] in self.methods:
        reject, pvalscorr = multipletests(pval0, alpha=self.alpha, method=val[1])[:2]
        assert_almost_equal(pvalscorr, res_multtest[:, val[0]], 15)
        assert_equal(reject, pvalscorr <= self.alpha)