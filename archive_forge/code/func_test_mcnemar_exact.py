from statsmodels.compat.python import lzip
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
from scipy import stats
import pytest
from statsmodels.stats.contingency_tables import (
from statsmodels.sandbox.stats.runs import (Runs,
from statsmodels.sandbox.stats.runs import mcnemar as sbmcnemar
from statsmodels.stats.nonparametric import (
from statsmodels.tools.testing import Holder
def test_mcnemar_exact():
    f_obs1 = np.array([[101, 121], [59, 33]])
    f_obs2 = np.array([[101, 70], [59, 33]])
    f_obs3 = np.array([[101, 80], [59, 33]])
    f_obs4 = np.array([[101, 30], [60, 33]])
    f_obs5 = np.array([[101, 10], [30, 33]])
    f_obs6 = np.array([[101, 10], [10, 33]])
    res1 = 4e-06
    res2 = 0.378688
    res3 = 0.089452
    res4 = 0.00206
    res5 = 0.002221
    res6 = 1.0
    stat = mcnemar(f_obs1, exact=True)
    assert_almost_equal([stat.statistic, stat.pvalue], [59, res1], decimal=6)
    stat = mcnemar(f_obs2, exact=True)
    assert_almost_equal([stat.statistic, stat.pvalue], [59, res2], decimal=6)
    stat = mcnemar(f_obs3, exact=True)
    assert_almost_equal([stat.statistic, stat.pvalue], [59, res3], decimal=6)
    stat = mcnemar(f_obs4, exact=True)
    assert_almost_equal([stat.statistic, stat.pvalue], [30, res4], decimal=6)
    stat = mcnemar(f_obs5, exact=True)
    assert_almost_equal([stat.statistic, stat.pvalue], [10, res5], decimal=6)
    stat = mcnemar(f_obs6, exact=True)
    assert_almost_equal([stat.statistic, stat.pvalue], [10, res6], decimal=6)