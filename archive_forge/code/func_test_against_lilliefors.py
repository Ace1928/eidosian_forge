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
def test_against_lilliefors(self):
    rng = np.random.default_rng(2291803665717442724)
    x = examgrades
    res = goodness_of_fit(stats.norm, x, statistic='ks', random_state=rng)
    known_params = {'loc': np.mean(x), 'scale': np.std(x, ddof=1)}
    ref = stats.kstest(x, stats.norm(**known_params).cdf, method='exact')
    assert_allclose(res.statistic, ref.statistic)
    assert_allclose(res.pvalue, 0.0348, atol=0.005)