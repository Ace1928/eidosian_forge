import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_ftest_power():
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5]:
        res0 = smp.ttest_power(0.01, 200, alpha)
        res1 = smp.ftest_power(0.01, 199, 1, alpha=alpha, ncc=0)
        assert_almost_equal(res1, res0, decimal=6)
    res1 = smp.ftest_anova_power(0.25, 200, 0.1592, k_groups=10)
    res0 = 0.8408
    assert_almost_equal(res1, res0, decimal=4)
    res2 = Holder()
    res2.u = 5
    res2.v = 199
    res2.f2 = 0.01
    res2.sig_level = 0.01
    res2.power = 0.0494137732920332
    res2.method = 'Multiple regression power calculation'
    res1 = smp.ftest_power(np.sqrt(res2.f2), res2.v, res2.u, alpha=res2.sig_level, ncc=1)
    assert_almost_equal(res1, res2.power, decimal=5)
    res2 = Holder()
    res2.u = 5
    res2.v = 199
    res2.f2 = 0.09
    res2.sig_level = 0.01
    res2.power = 0.7967191006290872
    res2.method = 'Multiple regression power calculation'
    res1 = smp.ftest_power(np.sqrt(res2.f2), res2.v, res2.u, alpha=res2.sig_level, ncc=1)
    assert_almost_equal(res1, res2.power, decimal=5)
    res2 = Holder()
    res2.u = 5
    res2.v = 19
    res2.f2 = 0.09
    res2.sig_level = 0.1
    res2.power = 0.235454222377575
    res2.method = 'Multiple regression power calculation'
    res1 = smp.ftest_power(np.sqrt(res2.f2), res2.v, res2.u, alpha=res2.sig_level, ncc=1)
    assert_almost_equal(res1, res2.power, decimal=5)