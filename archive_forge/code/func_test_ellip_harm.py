import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def test_ellip_harm():

    def E01(h2, k2, s):
        return 1

    def E11(h2, k2, s):
        return s

    def E12(h2, k2, s):
        return sqrt(abs(s * s - h2))

    def E13(h2, k2, s):
        return sqrt(abs(s * s - k2))

    def E21(h2, k2, s):
        return s * s - 1 / 3 * (h2 + k2 + sqrt(abs((h2 + k2) * (h2 + k2) - 3 * h2 * k2)))

    def E22(h2, k2, s):
        return s * s - 1 / 3 * (h2 + k2 - sqrt(abs((h2 + k2) * (h2 + k2) - 3 * h2 * k2)))

    def E23(h2, k2, s):
        return s * sqrt(abs(s * s - h2))

    def E24(h2, k2, s):
        return s * sqrt(abs(s * s - k2))

    def E25(h2, k2, s):
        return sqrt(abs((s * s - h2) * (s * s - k2)))

    def E31(h2, k2, s):
        return s * s * s - s / 5 * (2 * (h2 + k2) + sqrt(4 * (h2 + k2) * (h2 + k2) - 15 * h2 * k2))

    def E32(h2, k2, s):
        return s * s * s - s / 5 * (2 * (h2 + k2) - sqrt(4 * (h2 + k2) * (h2 + k2) - 15 * h2 * k2))

    def E33(h2, k2, s):
        return sqrt(abs(s * s - h2)) * (s * s - 1 / 5 * (h2 + 2 * k2 + sqrt(abs((h2 + 2 * k2) * (h2 + 2 * k2) - 5 * h2 * k2))))

    def E34(h2, k2, s):
        return sqrt(abs(s * s - h2)) * (s * s - 1 / 5 * (h2 + 2 * k2 - sqrt(abs((h2 + 2 * k2) * (h2 + 2 * k2) - 5 * h2 * k2))))

    def E35(h2, k2, s):
        return sqrt(abs(s * s - k2)) * (s * s - 1 / 5 * (2 * h2 + k2 + sqrt(abs((2 * h2 + k2) * (2 * h2 + k2) - 5 * h2 * k2))))

    def E36(h2, k2, s):
        return sqrt(abs(s * s - k2)) * (s * s - 1 / 5 * (2 * h2 + k2 - sqrt(abs((2 * h2 + k2) * (2 * h2 + k2) - 5 * h2 * k2))))

    def E37(h2, k2, s):
        return s * sqrt(abs((s * s - h2) * (s * s - k2)))
    assert_equal(ellip_harm(5, 8, 1, 2, 2.5, 1, 1), ellip_harm(5, 8, 1, 2, 2.5))
    known_funcs = {(0, 1): E01, (1, 1): E11, (1, 2): E12, (1, 3): E13, (2, 1): E21, (2, 2): E22, (2, 3): E23, (2, 4): E24, (2, 5): E25, (3, 1): E31, (3, 2): E32, (3, 3): E33, (3, 4): E34, (3, 5): E35, (3, 6): E36, (3, 7): E37}
    point_ref = []

    def ellip_harm_known(h2, k2, n, p, s):
        for i in range(h2.size):
            func = known_funcs[int(n[i]), int(p[i])]
            point_ref.append(func(h2[i], k2[i], s[i]))
        return point_ref
    np.random.seed(1234)
    h2 = np.random.pareto(0.5, size=30)
    k2 = h2 * (1 + np.random.pareto(0.5, size=h2.size))
    s = np.random.pareto(0.5, size=h2.size)
    points = []
    for i in range(h2.size):
        for n in range(4):
            for p in range(1, 2 * n + 2):
                points.append((h2[i], k2[i], n, p, s[i]))
    points = np.array(points)
    assert_func_equal(ellip_harm, ellip_harm_known, points, rtol=1e-12)