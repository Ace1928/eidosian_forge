from sympy.concrete.products import Product
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.trace import Trace
from sympy.tensor.indexed import IndexedBase
from sympy.stats import (GaussianUnitaryEnsemble as GUE, density,
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import RandomMatrixSymbol
from sympy.stats.random_matrix_models import GaussianEnsemble, RandomMatrixPSpace
from sympy.testing.pytest import raises
def test_GaussianOrthogonalEnsemble():
    H = RandomMatrixSymbol('H', 3, 3)
    _H = MatrixSymbol('_H', 3, 3)
    G = GOE('O', 3)
    assert density(G)(H) == exp(-3 * Trace(H ** 2) / 4) / Integral(exp(-3 * Trace(_H ** 2) / 4), _H)
    i, j = (Dummy('i', integer=True, positive=True), Dummy('j', integer=True, positive=True))
    l = IndexedBase('l')
    assert joint_eigen_distribution(G).dummy_eq(Lambda((l[1], l[2], l[3]), 9 * sqrt(2) * exp(-3 * l[1] ** 2 / 2 - 3 * l[2] ** 2 / 2 - 3 * l[3] ** 2 / 2) * Product(Abs(l[i] - l[j]), (j, i + 1, 3), (i, 1, 2)) / (32 * pi)))
    s = Dummy('s')
    assert level_spacing_distribution(G).dummy_eq(Lambda(s, s * pi * exp(-s ** 2 * pi / 4) / 2))