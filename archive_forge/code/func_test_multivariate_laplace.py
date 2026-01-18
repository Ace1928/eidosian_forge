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
def test_multivariate_laplace():
    raises(ValueError, lambda: Laplace('T', [1, 2], [[1, 2], [2, 1]]))
    L = Laplace('L', [1, 0], [[1, 0], [0, 1]])
    L2 = MultivariateLaplace('L2', [1, 0], [[1, 0], [0, 1]])
    assert density(L)(2, 3) == exp(2) * besselk(0, sqrt(39)) / pi
    L1 = Laplace('L1', [1, 2], [[x, 0], [0, y]])
    assert density(L1)(0, 1) == exp(2 / y) * besselk(0, sqrt((2 + 4 / y + 1 / x) / y)) / (pi * sqrt(x * y))
    assert L.pspace.distribution.set == ProductSet(S.Reals, S.Reals)
    assert L.pspace.distribution == L2.pspace.distribution