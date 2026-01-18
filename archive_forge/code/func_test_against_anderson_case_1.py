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
def test_against_anderson_case_1(self):
    rng = np.random.default_rng(5040212485680146248)
    x = np.arange(1, 101)
    known_params = {'scale': 29.957112639101933}
    res = goodness_of_fit(stats.norm, x, known_params=known_params, statistic='ad', random_state=rng)
    assert_allclose(res.statistic, 0.908)
    assert_allclose(res.pvalue, 0.1, atol=0.005)