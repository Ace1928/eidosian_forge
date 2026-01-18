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
def test_mcnemar_chisquare():
    f_obs1 = np.array([[101, 121], [59, 33]])
    f_obs2 = np.array([[101, 70], [59, 33]])
    f_obs3 = np.array([[101, 80], [59, 33]])
    res1 = [20.67222, 5.450095e-06]
    res2 = [0.7751938, 0.3786151]
    res3 = [2.87769784, 0.08981434]
    stat = mcnemar(f_obs1, exact=False)
    assert_allclose([stat.statistic, stat.pvalue], res1, rtol=1e-06)
    stat = mcnemar(f_obs2, exact=False)
    assert_allclose([stat.statistic, stat.pvalue], res2, rtol=1e-06)
    stat = mcnemar(f_obs3, exact=False)
    assert_allclose([stat.statistic, stat.pvalue], res3, rtol=1e-06)
    res1 = [21.35556, 3.815136e-06]
    res2 = [0.9379845, 0.3327967]
    res3 = [3.17266187, 0.07488031]
    res = mcnemar(f_obs1, exact=False, correction=False)
    assert_allclose([res.statistic, res.pvalue], res1, rtol=1e-06)
    res = mcnemar(f_obs2, exact=False, correction=False)
    assert_allclose([res.statistic, res.pvalue], res2, rtol=1e-06)
    res = mcnemar(f_obs3, exact=False, correction=False)
    assert_allclose([res.statistic, res.pvalue], res3, rtol=1e-06)