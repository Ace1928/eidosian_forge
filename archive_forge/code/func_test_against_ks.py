import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
def test_against_ks(self):
    rng = np.random.default_rng(8517426291317196949)
    x = examgrades
    known_params = {'loc': np.mean(x), 'scale': np.std(x, ddof=1)}
    res = goodness_of_fit(stats.norm, x, known_params=known_params, statistic='ks', random_state=rng)
    ref = stats.kstest(x, stats.norm(**known_params).cdf, method='exact')
    assert_allclose(res.statistic, ref.statistic)
    assert_allclose(res.pvalue, ref.pvalue, atol=0.005)