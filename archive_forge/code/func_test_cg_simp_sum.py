from sympy.concrete.summations import Sum
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.cg import Wigner3j, Wigner6j, Wigner9j, CG, cg_simp
from sympy.functions.special.tensor_functions import KroneckerDelta
def test_cg_simp_sum():
    x, a, b, c, cp, alpha, beta, gamma, gammap = symbols('x a b c cp alpha beta gamma gammap')
    assert cg_simp(x * Sum(CG(a, alpha, b, 0, a, alpha), (alpha, -a, a))) == x * (2 * a + 1) * KroneckerDelta(b, 0)
    assert cg_simp(x * Sum(CG(a, alpha, b, 0, a, alpha), (alpha, -a, a)) + CG(1, 0, 1, 0, 1, 0)) == x * (2 * a + 1) * KroneckerDelta(b, 0) + CG(1, 0, 1, 0, 1, 0)
    assert cg_simp(2 * Sum(CG(1, alpha, 0, 0, 1, alpha), (alpha, -1, 1))) == 6
    assert cg_simp(x * Sum((-1) ** (a - alpha) * CG(a, alpha, a, -alpha, c, 0), (alpha, -a, a))) == x * sqrt(2 * a + 1) * KroneckerDelta(c, 0)
    assert cg_simp(3 * Sum((-1) ** (2 - alpha) * CG(2, alpha, 2, -alpha, 0, 0), (alpha, -2, 2))) == 3 * sqrt(5)
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma) * CG(a, alpha, b, beta, cp, gammap), (alpha, -a, a), (beta, -b, b))) == KroneckerDelta(c, cp) * KroneckerDelta(gamma, gammap)
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma) * CG(a, alpha, b, beta, c, gammap), (alpha, -a, a), (beta, -b, b))) == KroneckerDelta(gamma, gammap)
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma) * CG(a, alpha, b, beta, cp, gamma), (alpha, -a, a), (beta, -b, b))) == KroneckerDelta(c, cp)
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma) ** 2, (alpha, -a, a), (beta, -b, b))) == 1
    assert cg_simp(Sum(CG(2, alpha, 1, beta, 2, gamma) * CG(2, alpha, 1, beta, 2, gammap), (alpha, -2, 2), (beta, -1, 1))) == KroneckerDelta(gamma, gammap)