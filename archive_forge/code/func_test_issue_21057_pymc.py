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
@XFAIL
def test_issue_21057_pymc():
    m = Normal('x', [0, 0], [[0, 0], [0, 0]])
    n = MultivariateNormal('x', [0, 0], [[0, 0], [0, 0]])
    p = Normal('x', [0, 0], [[0, 0], [0, 1]])
    assert m == n
    libraries = ('pymc',)
    for library in libraries:
        try:
            imported_lib = import_module(library)
            if imported_lib:
                s1 = sample(m, size=8, library=library)
                s2 = sample(n, size=8, library=library)
                s3 = sample(p, size=8, library=library)
                assert tuple(s1.flatten()) == tuple(s2.flatten())
                for s in s3:
                    assert tuple(s.flatten()) in p.pspace.distribution.set
        except NotImplementedError:
            continue