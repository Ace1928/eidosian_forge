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
def test_binomial_numeric():
    nvals = range(5)
    pvals = [0, Rational(1, 4), S.Half, Rational(3, 4), 1]
    for n in nvals:
        for p in pvals:
            X = Binomial('X', n, p)
            assert E(X) == n * p
            assert variance(X) == n * p * (1 - p)
            if n > 0 and 0 < p < 1:
                assert skewness(X) == (1 - 2 * p) / sqrt(n * p * (1 - p))
                assert kurtosis(X) == 3 + (1 - 6 * p * (1 - p)) / (n * p * (1 - p))
            for k in range(n + 1):
                assert P(Eq(X, k)) == binomial(n, k) * p ** k * (1 - p) ** (n - k)