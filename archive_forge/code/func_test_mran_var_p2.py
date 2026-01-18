import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_mran_var_p2(self):
    n, p = zinegbin.convert_params(7, 1, 2)
    nbinom_mean, nbinom_var = (nbinom.mean(n, p), nbinom.var(n, p))
    zinb_mean = zinegbin.mean(7, 1, 2, 0)
    zinb_var = zinegbin.var(7, 1, 2, 0)
    assert_allclose(nbinom_mean, zinb_mean, rtol=1e-10)
    assert_allclose(nbinom_var, zinb_var, rtol=1e-10)