from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_gamma_quotients():
    mp.dps = 15
    h = 1e-08
    ep = 0.0001
    G = gamma
    assert gammaprod([-1], [-3, -4]) == 0
    assert gammaprod([-1, 0], [-5]) == inf
    assert abs(gammaprod([-1], [-2]) - G(-1 + h) / G(-2 + h)) < 0.0001
    assert abs(gammaprod([-4, -3], [-2, 0]) - G(-4 + h) * G(-3 + h) / G(-2 + h) / G(0 + h)) < 0.0001
    assert rf(3, 0) == 1
    assert rf(2.5, 1) == 2.5
    assert rf(-5, 2) == 20
    assert rf(j, j).ae(gamma(2 * j) / gamma(j))
    assert rf('-255.5815971722918', '-0.5119253100282322').ae('-0.1952720278805729485')
    assert ff(-2, 0) == 1
    assert ff(-2, 1) == -2
    assert ff(4, 3) == 24
    assert ff(3, 4) == 0
    assert binomial(0, 0) == 1
    assert binomial(1, 0) == 1
    assert binomial(0, -1) == 0
    assert binomial(3, 2) == 3
    assert binomial(5, 2) == 10
    assert binomial(5, 3) == 10
    assert binomial(5, 5) == 1
    assert binomial(-1, 0) == 1
    assert binomial(-2, -4) == 3
    assert binomial(4.5, 1.5) == 6.5625
    assert binomial(1100, 1) == 1100
    assert binomial(1100, 2) == 604450
    assert beta(1, 1) == 1
    assert beta(0, 0) == inf
    assert beta(3, 0) == inf
    assert beta(-1, -1) == inf
    assert beta(1.5, 1).ae(2 / 3.0)
    assert beta(1.5, 2.5).ae(pi / 16)
    assert (10 ** 15 * beta(10, 100)).ae(2.345533973960465)
    assert beta(inf, inf) == 0
    assert isnan(beta(-inf, inf))
    assert isnan(beta(-3, inf))
    assert isnan(beta(0, inf))
    assert beta(inf, 0.5) == beta(0.5, inf) == 0
    assert beta(inf, -1.5) == inf
    assert beta(inf, -0.5) == -inf
    assert beta(1 + 2j, -1 - j / 2).ae(1.1639654245106994 + 0.08511695947832915j)
    assert beta(-0.5, 0.5) == 0
    assert beta(-3, 3).ae(-1 / 3.0)
    assert beta('-255.5815971722918', '-0.5119253100282322').ae('18.157330562703710339')