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
def test_binomial_quantile():
    X = Binomial('X', 50, S.Half)
    assert quantile(X)(0.95) == S(31)
    assert median(X) == FiniteSet(25)
    X = Binomial('X', 5, S.Half)
    p = Symbol('p', positive=True)
    assert quantile(X)(p) == Piecewise((nan, p > S.One), (S.Zero, p <= Rational(1, 32)), (S.One, p <= Rational(3, 16)), (S(2), p <= S.Half), (S(3), p <= Rational(13, 16)), (S(4), p <= Rational(31, 32)), (S(5), p <= S.One))
    assert median(X) == FiniteSet(2, 3)