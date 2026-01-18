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
@pytest.mark.slow
def test_against_anderson_gumbel_r(self):
    rng = np.random.default_rng(7302761058217743)
    x = stats.genextreme(0.051896837188595134, loc=0.5, scale=1.5).rvs(size=1000, random_state=rng)
    res = goodness_of_fit(stats.gumbel_r, x, statistic='ad', random_state=rng)
    ref = stats.anderson(x, dist='gumbel_r')
    assert_allclose(res.statistic, ref.critical_values[0])
    assert_allclose(res.pvalue, ref.significance_level[0] / 100, atol=0.005)