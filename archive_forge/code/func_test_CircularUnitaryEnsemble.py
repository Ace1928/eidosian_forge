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
def test_CircularUnitaryEnsemble():
    CU = CUE('U', 3)
    j, k = (Dummy('j', integer=True, positive=True), Dummy('k', integer=True, positive=True))
    t = IndexedBase('t')
    assert joint_eigen_distribution(CU).dummy_eq(Lambda((t[1], t[2], t[3]), Product(Abs(exp(I * t[j]) - exp(I * t[k])) ** 2, (j, k + 1, 3), (k, 1, 2)) / (48 * pi ** 3)))