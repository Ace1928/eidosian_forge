import copy
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.stats._multicomp import _pvalue_dunnett, DunnettResult
@pytest.mark.parametrize('rho, n_groups, df, statistic, pvalue, alternative', [(0.5, 1, 10, 1.81, 0.05, 'greater'), (0.5, 3, 10, 2.34, 0.05, 'greater'), (0.5, 2, 30, 1.99, 0.05, 'greater'), (0.5, 5, 30, 2.33, 0.05, 'greater'), (0.5, 4, 12, 3.32, 0.01, 'greater'), (0.5, 7, 12, 3.56, 0.01, 'greater'), (0.5, 2, 60, 2.64, 0.01, 'greater'), (0.5, 4, 60, 2.87, 0.01, 'greater'), (0.5, 4, 60, [2.87, 2.21], [0.01, 0.05], 'greater'), (0.5, 1, 10, 2.23, 0.05, 'two-sided'), (0.5, 3, 10, 2.81, 0.05, 'two-sided'), (0.5, 2, 30, 2.32, 0.05, 'two-sided'), (0.5, 3, 20, 2.57, 0.05, 'two-sided'), (0.5, 4, 12, 3.76, 0.01, 'two-sided'), (0.5, 7, 12, 4.08, 0.01, 'two-sided'), (0.5, 2, 60, 2.9, 0.01, 'two-sided'), (0.5, 4, 60, 3.14, 0.01, 'two-sided'), (0.5, 4, 60, [3.14, 2.55], [0.01, 0.05], 'two-sided')])
def test_critical_values(self, rho, n_groups, df, statistic, pvalue, alternative):
    rng = np.random.default_rng(165250594791731684851746311027739134893)
    rho = np.full((n_groups, n_groups), rho)
    np.fill_diagonal(rho, 1)
    statistic = np.array(statistic)
    res = _pvalue_dunnett(rho=rho, df=df, statistic=statistic, alternative=alternative, rng=rng)
    assert_allclose(res, pvalue, atol=0.005)