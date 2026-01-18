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
def test_hypergeometric_symbolic():
    N, m, n = symbols('N, m, n')
    H = Hypergeometric('H', N, m, n)
    dens = density(H).dict
    expec = E(H > 2)
    assert dens == Density(HypergeometricDistribution(N, m, n))
    assert dens.subs(N, 5).doit() == Density(HypergeometricDistribution(5, m, n))
    assert set(dens.subs({N: 3, m: 2, n: 1}).doit().keys()) == {S.Zero, S.One}
    assert set(dens.subs({N: 3, m: 2, n: 1}).doit().values()) == {Rational(1, 3), Rational(2, 3)}
    k = Dummy('k', integer=True)
    assert expec.dummy_eq(Sum(Piecewise((k * binomial(m, k) * binomial(N - m, -k + n) / binomial(N, n), k > 2), (0, True)), (k, 0, n)))