import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
def test_equivalence_welch(self):
    means = self.means
    nobs = self.nobs
    stds = self.stds
    n_groups = self.n_groups
    vars_ = stds ** 2
    eps = 0.5
    res0 = anova_generic(means, vars_, nobs, use_var='unequal', welch_correction=False)
    f_stat = res0.statistic
    res = equivalence_oneway_generic(f_stat, n_groups, nobs.sum(), eps, res0.df, alpha=0.05, margin_type='wellek')
    assert_allclose(res.pvalue, 0.011, atol=0.001)
    assert_allclose(res.df, [3.0, 22.6536], atol=0.0006)
    assert_allclose(f_stat, 0.1102, atol=0.007)
    res = equivalence_oneway(self.data, eps, use_var='unequal', margin_type='wellek')
    assert_allclose(res.pvalue, 0.011, atol=0.0001)
    assert_allclose(res.df, [3.0, 22.6536], atol=0.0006)
    assert_allclose(res.f_stat, 0.1102, atol=0.0001)
    pow_ = _power_equivalence_oneway_emp(f_stat, n_groups, nobs, eps, res0.df)
    assert_allclose(pow_, 0.1552, atol=0.007)
    pow_ = power_equivalence_oneway(eps, eps, nobs.sum(), n_groups=n_groups, df=None, alpha=0.05, margin_type='wellek')
    assert_allclose(pow_, 0.05, atol=1e-13)
    nobs_t = nobs.sum()
    es = effectsize_oneway(means, vars_, nobs, use_var='unequal')
    es = np.sqrt(es)
    es_w0 = f2_to_wellek(es ** 2, n_groups)
    es_w = np.sqrt(fstat_to_wellek(f_stat, n_groups, nobs_t / n_groups))
    pow_ = power_equivalence_oneway(es_w, eps, nobs_t, n_groups=n_groups, df=None, alpha=0.05, margin_type='wellek')
    assert_allclose(pow_, 0.1552, atol=0.007)
    assert_allclose(es_w0, es_w, atol=0.007)
    margin = wellek_to_f2(eps, n_groups)
    pow_ = power_equivalence_oneway(es ** 2, margin, nobs_t, n_groups=n_groups, df=None, alpha=0.05, margin_type='f2')
    assert_allclose(pow_, 0.1552, atol=0.007)