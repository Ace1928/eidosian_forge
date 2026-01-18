from sympy.concrete.summations import Sum
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, nan)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.beta_functions import beta
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polytools import cancel
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.matrices import Matrix
from sympy.stats import (DiscreteUniform, Die, Bernoulli, Coin, Binomial, BetaBinomial,
from sympy.stats.frv_types import DieDistribution, BinomialDistribution, \
from sympy.stats.rv import Density
from sympy.testing.pytest import raises
def test_binomial_symbolic():
    n = 2
    p = symbols('p', positive=True)
    X = Binomial('X', n, p)
    t = Symbol('t')
    assert simplify(E(X)) == n * p == simplify(moment(X, 1))
    assert simplify(variance(X)) == n * p * (1 - p) == simplify(cmoment(X, 2))
    assert cancel(skewness(X) - (1 - 2 * p) / sqrt(n * p * (1 - p))) == 0
    assert cancel(kurtosis(X) - (3 + (1 - 6 * p * (1 - p)) / (n * p * (1 - p)))) == 0
    assert characteristic_function(X)(t) == p ** 2 * exp(2 * I * t) + 2 * p * (-p + 1) * exp(I * t) + (-p + 1) ** 2
    assert moment_generating_function(X)(t) == p ** 2 * exp(2 * t) + 2 * p * (-p + 1) * exp(t) + (-p + 1) ** 2
    H, T = symbols('H T')
    Y = Binomial('Y', n, p, succ=H, fail=T)
    assert simplify(E(Y) - n * (H * p + T * (1 - p))) == 0
    n = symbols('n')
    B = Binomial('B', n, p)
    raises(NotImplementedError, lambda: P(B > 2))
    assert density(B).dict == Density(BinomialDistribution(n, p, 1, 0))
    assert set(density(B).dict.subs(n, 4).doit().keys()) == {S.Zero, S.One, S(2), S(3), S(4)}
    assert set(density(B).dict.subs(n, 4).doit().values()) == {(1 - p) ** 4, 4 * p * (1 - p) ** 3, 6 * p ** 2 * (1 - p) ** 2, 4 * p ** 3 * (1 - p), p ** 4}
    k = Dummy('k', integer=True)
    assert E(B > 2).dummy_eq(Sum(Piecewise((k * p ** k * (1 - p) ** (-k + n) * binomial(n, k), (k >= 0) & (k <= n) & (k > 2)), (0, True)), (k, 0, n)))