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
def test_against_anderson_case_0(self):
    rng = np.random.default_rng(7384539336846690410)
    x = np.arange(1, 101)
    known_params = {'loc': 45.01575354024957, 'scale': 30}
    res = goodness_of_fit(stats.norm, x, known_params=known_params, statistic='ad', random_state=rng)
    assert_allclose(res.statistic, 2.492)
    assert_allclose(res.pvalue, 0.05, atol=0.005)