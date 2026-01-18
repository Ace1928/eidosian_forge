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
def test_moment_generating_functions():
    t = S('t')
    geometric_mgf = moment_generating_function(Geometric('g', S.Half))(t)
    assert geometric_mgf.diff(t).subs(t, 0) == 2
    logarithmic_mgf = moment_generating_function(Logarithmic('l', S.Half))(t)
    assert logarithmic_mgf.diff(t).subs(t, 0) == 1 / log(2)
    negative_binomial_mgf = moment_generating_function(NegativeBinomial('n', 5, Rational(1, 3)))(t)
    assert negative_binomial_mgf.diff(t).subs(t, 0) == Rational(5, 2)
    poisson_mgf = moment_generating_function(Poisson('p', 5))(t)
    assert poisson_mgf.diff(t).subs(t, 0) == 5
    skellam_mgf = moment_generating_function(Skellam('s', 1, 1))(t)
    assert skellam_mgf.diff(t).subs(t, 2) == (-exp(-2) + exp(2)) * exp(-2 + exp(-2) + exp(2))
    yule_simon_mgf = moment_generating_function(YuleSimon('y', 3))(t)
    assert simplify(yule_simon_mgf.diff(t).subs(t, 0)) == Rational(3, 2)
    zeta_mgf = moment_generating_function(Zeta('z', 5))(t)
    assert zeta_mgf.diff(t).subs(t, 0) == pi ** 4 / (90 * zeta(5))