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
@pytest.mark.parametrize('case', [(25, [0.928, 0.937, 0.95, 0.958, 0.966]), (50, [0.959, 0.965, 0.972, 0.977, 0.981]), (95, [0.977, 0.979, 0.983, 0.986, 0.989])])
def test_against_filliben_norm_table(self, case):
    rng = np.random.default_rng(504569995557928957)
    n, ref = case
    x = rng.random(n)
    known_params = {'loc': 0, 'scale': 1}
    res = stats.goodness_of_fit(stats.norm, x, known_params=known_params, statistic='filliben', random_state=rng)
    percentiles = np.array([0.005, 0.01, 0.025, 0.05, 0.1])
    res = stats.scoreatpercentile(res.null_distribution, percentiles * 100)
    assert_allclose(res, ref, atol=0.002)