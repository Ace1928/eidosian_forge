from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.external import import_module
from sympy.stats import Binomial, sample, Die, FiniteRV, DiscreteUniform, Bernoulli, BetaBinomial, Hypergeometric, \
from sympy.testing.pytest import skip, raises
def test_given_sample():
    X = Die('X', 6)
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    assert sample(X, X > 5) == 6