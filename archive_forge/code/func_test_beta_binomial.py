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
def test_beta_binomial():
    raises(ValueError, lambda: BetaBinomial('b', 0.2, 1, 2))
    raises(ValueError, lambda: BetaBinomial('b', 2, -1, 2))
    raises(ValueError, lambda: BetaBinomial('b', 2, 1, -2))
    assert BetaBinomial('b', 2, 1, 1)
    nvals = range(1, 5)
    alphavals = [Rational(1, 4), S.Half, Rational(3, 4), 1, 10]
    betavals = [Rational(1, 4), S.Half, Rational(3, 4), 1, 10]
    for n in nvals:
        for a in alphavals:
            for b in betavals:
                X = BetaBinomial('X', n, a, b)
                assert E(X) == moment(X, 1)
                assert variance(X) == cmoment(X, 2)
    n, a, b = symbols('a b n')
    assert BetaBinomial('x', n, a, b)
    n = 2
    a, b = symbols('a b', positive=True)
    X = BetaBinomial('X', n, a, b)
    t = Symbol('t')
    assert E(X).expand() == moment(X, 1).expand()
    assert variance(X).expand() == cmoment(X, 2).expand()
    assert skewness(X) == smoment(X, 3)
    assert characteristic_function(X)(t) == exp(2 * I * t) * beta(a + 2, b) / beta(a, b) + 2 * exp(I * t) * beta(a + 1, b + 1) / beta(a, b) + beta(a, b + 2) / beta(a, b)
    assert moment_generating_function(X)(t) == exp(2 * t) * beta(a + 2, b) / beta(a, b) + 2 * exp(t) * beta(a + 1, b + 1) / beta(a, b) + beta(a, b + 2) / beta(a, b)