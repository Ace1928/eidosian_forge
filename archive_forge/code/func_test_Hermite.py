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
def test_Hermite():
    a1 = Symbol('a1', positive=True)
    a2 = Symbol('a2', negative=True)
    raises(ValueError, lambda: Hermite('H', a1, a2))
    a1 = Symbol('a1', negative=True)
    a2 = Symbol('a2', positive=True)
    raises(ValueError, lambda: Hermite('H', a1, a2))
    a1 = Symbol('a1', positive=True)
    x = Symbol('x')
    H = Hermite('H', a1, a2)
    assert moment_generating_function(H)(x) == exp(a1 * (exp(x) - 1) + a2 * (exp(2 * x) - 1))
    assert characteristic_function(H)(x) == exp(a1 * (exp(I * x) - 1) + a2 * (exp(2 * I * x) - 1))
    assert E(H) == a1 + 2 * a2
    H = Hermite('H', a1=5, a2=4)
    assert density(H)(2) == 33 * exp(-9) / 2
    assert E(H) == 13
    assert variance(H) == 21
    assert kurtosis(H) == Rational(464, 147)
    assert skewness(H) == 37 * sqrt(21) / 441