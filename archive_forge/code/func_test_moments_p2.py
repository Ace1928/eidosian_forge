import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_moments_p2(self):
    n, p = zinegbin.convert_params(7, 1, 2)
    nb_m1, nb_m2 = (nbinom.moment(1, n, p), nbinom.moment(2, n, p))
    zinb_m0 = zinegbin.moment(0, 7, 1, 2, 0)
    zinb_m1 = zinegbin.moment(1, 7, 1, 2, 0)
    zinb_m2 = zinegbin.moment(2, 7, 1, 2, 0)
    assert_allclose(1, zinb_m0, rtol=1e-10)
    assert_allclose(nb_m1, zinb_m1, rtol=1e-10)
    assert_allclose(nb_m2, zinb_m2, rtol=1e-10)