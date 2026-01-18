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
def test_conditional():
    X = Geometric('X', Rational(2, 3))
    Y = Poisson('Y', 3)
    assert P(X > 2, X > 3) == 1
    assert P(X > 3, X > 2) == Rational(1, 3)
    assert P(Y > 2, Y < 2) == 0
    assert P(Eq(Y, 3), Y >= 0) == 9 * exp(-3) / 2
    assert P(Eq(Y, 3), Eq(Y, 2)) == 0
    assert P(X < 2, Eq(X, 2)) == 0
    assert P(X > 2, Eq(X, 3)) == 1