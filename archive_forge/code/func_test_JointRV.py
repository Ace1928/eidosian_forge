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
def test_JointRV():
    x1, x2 = (Indexed('x', i) for i in (1, 2))
    pdf = exp(-x1 ** 2 / 2 + x1 - x2 ** 2 / 2 - S.Half) / (2 * pi)
    X = JointRV('x', pdf)
    assert density(X)(1, 2) == exp(-2) / (2 * pi)
    assert isinstance(X.pspace.distribution, JointDistributionHandmade)
    assert marginal_distribution(X, 0)(2) == sqrt(2) * exp(Rational(-1, 2)) / (2 * sqrt(pi))