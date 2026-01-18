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
def test_discreteuniform():
    a, b, c, t = symbols('a b c t')
    X = DiscreteUniform('X', [a, b, c])
    assert E(X) == (a + b + c) / 3
    assert simplify(variance(X) - ((a ** 2 + b ** 2 + c ** 2) / 3 - (a / 3 + b / 3 + c / 3) ** 2)) == 0
    assert P(Eq(X, a)) == P(Eq(X, b)) == P(Eq(X, c)) == S('1/3')
    Y = DiscreteUniform('Y', range(-5, 5))
    assert E(Y) == S('-1/2')
    assert variance(Y) == S('33/4')
    assert median(Y) == FiniteSet(-1, 0)
    for x in range(-5, 5):
        assert P(Eq(Y, x)) == S('1/10')
        assert P(Y <= x) == S(x + 6) / 10
        assert P(Y >= x) == S(5 - x) / 10
    assert dict(density(Die('D', 6)).items()) == dict(density(DiscreteUniform('U', range(1, 7))).items())
    assert characteristic_function(X)(t) == exp(I * a * t) / 3 + exp(I * b * t) / 3 + exp(I * c * t) / 3
    assert moment_generating_function(X)(t) == exp(a * t) / 3 + exp(b * t) / 3 + exp(c * t) / 3
    raises(ValueError, lambda: DiscreteUniform('Z', [a, a, a, b, b, c]))