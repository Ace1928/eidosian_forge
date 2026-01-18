from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
from sympy.functions.elementary.complexes import polar_lift
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besselk
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices.dense import eye
from sympy.matrices.expressions.determinant import Determinant
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Interval, ProductSet)
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.core.numbers import comp
from sympy.integrals.integrals import integrate
from sympy.matrices import Matrix, MatrixSymbol
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.stats import density, median, marginal_distribution, Normal, Laplace, E, sample
from sympy.stats.joint_rv_types import (JointRV, MultivariateNormalDistribution,
from sympy.testing.pytest import raises, XFAIL, skip, slow
from sympy.external import import_module
from sympy.abc import x, y
def test_NormalGamma():
    ng = NormalGamma('G', 1, 2, 3, 4)
    assert density(ng)(1, 1) == 32 * exp(-4) / sqrt(pi)
    assert ng.pspace.distribution.set == ProductSet(S.Reals, Interval(0, oo))
    raises(ValueError, lambda: NormalGamma('G', 1, 2, 3, -1))
    assert marginal_distribution(ng, 0)(1) == 3 * sqrt(10) * gamma(Rational(7, 4)) / (10 * sqrt(pi) * gamma(Rational(5, 4)))
    assert marginal_distribution(ng, y)(1) == exp(Rational(-1, 4)) / 128
    assert marginal_distribution(ng, [0, 1])(x) == x ** 2 * exp(-x / 4) / 128