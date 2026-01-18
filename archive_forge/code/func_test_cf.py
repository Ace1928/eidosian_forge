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
def test_cf(dist, support_lower_limit, support_upper_limit):
    pdf = density(dist)
    t = S('t')
    x = S('x')
    cf1 = lambdify([t], characteristic_function(dist)(t), 'mpmath')
    f = lambdify([x, t], pdf(x) * exp(I * x * t), 'mpmath')
    cf2 = lambda t: mpmath.nsum(lambda x: f(x, t), [support_lower_limit, support_upper_limit], maxdegree=10)
    for test_point in [2, 5, 8, 11]:
        n1 = cf1(test_point)
        n2 = cf2(test_point)
        assert abs(re(n1) - re(n2)) < 1e-12
        assert abs(im(n1) - im(n2)) < 1e-12