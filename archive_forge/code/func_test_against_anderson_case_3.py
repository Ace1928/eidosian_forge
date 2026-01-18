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
def test_against_anderson_case_3(self):
    rng = np.random.default_rng(6763691329830218206)
    x = stats.skewnorm.rvs(1.4477847789132101, loc=1, scale=2, size=100, random_state=rng)
    res = goodness_of_fit(stats.norm, x, statistic='ad', random_state=rng)
    assert_allclose(res.statistic, 0.559)
    assert_allclose(res.pvalue, 0.15, atol=0.005)