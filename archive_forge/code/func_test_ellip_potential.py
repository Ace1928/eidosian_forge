import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def test_ellip_potential():

    def change_coefficient(lambda1, mu, nu, h2, k2):
        x = sqrt(lambda1 ** 2 * mu ** 2 * nu ** 2 / (h2 * k2))
        y = sqrt((lambda1 ** 2 - h2) * (mu ** 2 - h2) * (h2 - nu ** 2) / (h2 * (k2 - h2)))
        z = sqrt((lambda1 ** 2 - k2) * (k2 - mu ** 2) * (k2 - nu ** 2) / (k2 * (k2 - h2)))
        return (x, y, z)

    def solid_int_ellip(lambda1, mu, nu, n, p, h2, k2):
        return ellip_harm(h2, k2, n, p, lambda1) * ellip_harm(h2, k2, n, p, mu) * ellip_harm(h2, k2, n, p, nu)

    def solid_int_ellip2(lambda1, mu, nu, n, p, h2, k2):
        return ellip_harm_2(h2, k2, n, p, lambda1) * ellip_harm(h2, k2, n, p, mu) * ellip_harm(h2, k2, n, p, nu)

    def summation(lambda1, mu1, nu1, lambda2, mu2, nu2, h2, k2):
        tol = 1e-08
        sum1 = 0
        for n in range(20):
            xsum = 0
            for p in range(1, 2 * n + 2):
                xsum += 4 * pi * (solid_int_ellip(lambda2, mu2, nu2, n, p, h2, k2) * solid_int_ellip2(lambda1, mu1, nu1, n, p, h2, k2)) / (ellip_normal(h2, k2, n, p) * (2 * n + 1))
            if abs(xsum) < 0.1 * tol * abs(sum1):
                break
            sum1 += xsum
        return (sum1, xsum)

    def potential(lambda1, mu1, nu1, lambda2, mu2, nu2, h2, k2):
        x1, y1, z1 = change_coefficient(lambda1, mu1, nu1, h2, k2)
        x2, y2, z2 = change_coefficient(lambda2, mu2, nu2, h2, k2)
        res = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return 1 / res
    pts = [(120, sqrt(19), 2, 41, sqrt(17), 2, 15, 25), (120, sqrt(16), 3.2, 21, sqrt(11), 2.9, 11, 20)]
    with suppress_warnings() as sup:
        sup.filter(IntegrationWarning, 'The occurrence of roundoff error')
        sup.filter(IntegrationWarning, 'The maximum number of subdivisions')
        for p in pts:
            err_msg = repr(p)
            exact = potential(*p)
            result, last_term = summation(*p)
            assert_allclose(exact, result, atol=0, rtol=1e-08, err_msg=err_msg)
            assert_(abs(result - exact) < 10 * abs(last_term), err_msg)