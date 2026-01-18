from sympy.concrete.summations import Sum
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besseli
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.zeta_functions import zeta
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.utilities.lambdify import lambdify
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.exponential import exp
from sympy.logic.boolalg import Or
from sympy.sets.fancysets import Range
from sympy.stats import (P, E, variance, density, characteristic_function,
from sympy.stats.drv_types import (PoissonDistribution, GeometricDistribution,
from sympy.testing.pytest import slow, nocache_fail, raises
from sympy.stats.symbolic_probability import Expectation
@slow
def test_GeometricDistribution():
    p = S.One / 5
    d = GeometricDistribution(p)
    assert d.expectation(x, x) == 1 / p
    assert d.expectation(x ** 2, x) - d.expectation(x, x) ** 2 == (1 - p) / p ** 2
    assert abs(d.cdf(20000).evalf() - 1) < 0.001
    assert abs(d.cdf(20000.8).evalf() - 1) < 0.001
    G = Geometric('G', p=S(1) / 4)
    assert cdf(G)(S(7) / 2) == P(G <= S(7) / 2)
    X = Geometric('X', Rational(1, 5))
    Y = Geometric('Y', Rational(3, 10))
    assert coskewness(X, X + Y, X + 2 * Y).simplify() == sqrt(230) * Rational(81, 1150)