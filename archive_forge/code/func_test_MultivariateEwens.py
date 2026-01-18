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
def test_MultivariateEwens():
    n, theta, i = symbols('n theta i', positive=True)
    theta_f = symbols('t_f', negative=True)
    a = symbols('a_1:4', positive=True, integer=True)
    ed = MultivariateEwens('E', 3, theta)
    assert density(ed)(a[0], a[1], a[2]) == Piecewise((6 * 2 ** (-a[1]) * 3 ** (-a[2]) * theta ** a[0] * theta ** a[1] * theta ** a[2] / (theta * (theta + 1) * (theta + 2) * factorial(a[0]) * factorial(a[1]) * factorial(a[2])), Eq(a[0] + 2 * a[1] + 3 * a[2], 3)), (0, True))
    assert marginal_distribution(ed, ed[1])(a[1]) == Piecewise((6 * 2 ** (-a[1]) * theta ** a[1] / ((theta + 1) * (theta + 2) * factorial(a[1])), Eq(2 * a[1] + 1, 3)), (0, True))
    raises(ValueError, lambda: MultivariateEwens('e1', 5, theta_f))
    assert ed.pspace.distribution.set == ProductSet(Range(0, 4, 1), Range(0, 2, 1), Range(0, 2, 1))
    eds = MultivariateEwens('E', n, theta)
    a = IndexedBase('a')
    j, k = symbols('j, k')
    den = Piecewise((factorial(n) * Product(theta ** a[j] * (j + 1) ** (-a[j]) / factorial(a[j]), (j, 0, n - 1)) / RisingFactorial(theta, n), Eq(n, Sum((k + 1) * a[k], (k, 0, n - 1)))), (0, True))
    assert density(eds)(a).dummy_eq(den)