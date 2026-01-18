import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_cdf_zero(self):
    poisson_cdf = poisson.cdf(3, 2)
    zipoisson_cdf = zipoisson.cdf(3, 2, 0)
    assert_allclose(poisson_cdf, zipoisson_cdf, rtol=1e-12)