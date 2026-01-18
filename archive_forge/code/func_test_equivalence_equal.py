import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
def test_equivalence_equal(self):
    means = self.means
    nobs = self.nobs
    stds = self.stds
    n_groups = self.n_groups
    eps = 0.5
    res0 = anova_generic(means, stds ** 2, nobs, use_var='equal')
    f = res0.statistic
    res = equivalence_oneway_generic(f, n_groups, nobs.sum(), eps, res0.df, alpha=0.05, margin_type='wellek')
    assert_allclose(res.pvalue, 0.0083, atol=0.001)
    assert_equal(res.df, [3, 46])
    assert_allclose(f, 0.0926, atol=0.0006)
    res = equivalence_oneway(self.data, eps, use_var='equal', margin_type='wellek')
    assert_allclose(res.pvalue, 0.0083, atol=0.001)
    assert_equal(res.df, [3, 46])