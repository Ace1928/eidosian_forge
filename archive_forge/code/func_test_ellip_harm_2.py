import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def test_ellip_harm_2():

    def I1(h2, k2, s):
        res = ellip_harm_2(h2, k2, 1, 1, s) / (3 * ellip_harm(h2, k2, 1, 1, s)) + ellip_harm_2(h2, k2, 1, 2, s) / (3 * ellip_harm(h2, k2, 1, 2, s)) + ellip_harm_2(h2, k2, 1, 3, s) / (3 * ellip_harm(h2, k2, 1, 3, s))
        return res
    with suppress_warnings() as sup:
        sup.filter(IntegrationWarning, 'The occurrence of roundoff error')
        assert_almost_equal(I1(5, 8, 10), 1 / (10 * sqrt((100 - 5) * (100 - 8))))
        assert_almost_equal(ellip_harm_2(5, 8, 2, 1, 10), 0.00108056853382)
        assert_almost_equal(ellip_harm_2(5, 8, 2, 2, 10), 0.00105820513809)
        assert_almost_equal(ellip_harm_2(5, 8, 2, 3, 10), 0.00106058384743)
        assert_almost_equal(ellip_harm_2(5, 8, 2, 4, 10), 0.00106774492306)
        assert_almost_equal(ellip_harm_2(5, 8, 2, 5, 10), 0.00107976356454)