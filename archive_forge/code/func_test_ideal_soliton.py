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
def test_ideal_soliton():
    raises(ValueError, lambda: IdealSoliton('sol', -12))
    raises(ValueError, lambda: IdealSoliton('sol', 13.2))
    raises(ValueError, lambda: IdealSoliton('sol', 0))
    f = Function('f')
    raises(ValueError, lambda: density(IdealSoliton('sol', 10)).pmf(f))
    k = Symbol('k', integer=True, positive=True)
    x = Symbol('x', integer=True, positive=True)
    t = Symbol('t')
    sol = IdealSoliton('sol', k)
    assert density(sol).low == S.One
    assert density(sol).high == k
    assert density(sol).dict == Density(density(sol))
    assert density(sol).pmf(x) == Piecewise((1 / k, Eq(x, 1)), (1 / (x * (x - 1)), k >= x), (0, True))
    k_vals = [5, 20, 50, 100, 1000]
    for i in k_vals:
        assert E(sol.subs(k, i)) == harmonic(i) == moment(sol.subs(k, i), 1)
        assert variance(sol.subs(k, i)) == i - 1 + harmonic(i) - harmonic(i) ** 2 == cmoment(sol.subs(k, i), 2)
        assert skewness(sol.subs(k, i)) == smoment(sol.subs(k, i), 3)
        assert kurtosis(sol.subs(k, i)) == smoment(sol.subs(k, i), 4)
    assert exp(I * t) / 10 + Sum(exp(I * t * x) / (x * x - x), (x, 2, k)).subs(k, 10).doit() == characteristic_function(sol.subs(k, 10))(t)
    assert exp(t) / 10 + Sum(exp(t * x) / (x * x - x), (x, 2, k)).subs(k, 10).doit() == moment_generating_function(sol.subs(k, 10))(t)