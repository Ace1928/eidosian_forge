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
def test_JointEigenDistribution():
    A = Matrix([[Normal('A00', 0, 1), Normal('A01', 1, 1)], [Beta('A10', 1, 1), Beta('A11', 1, 1)]])
    assert JointEigenDistribution(A) == JointDistributionHandmade(-sqrt(A[0, 0] ** 2 - 2 * A[0, 0] * A[1, 1] + 4 * A[0, 1] * A[1, 0] + A[1, 1] ** 2) / 2 + A[0, 0] / 2 + A[1, 1] / 2, sqrt(A[0, 0] ** 2 - 2 * A[0, 0] * A[1, 1] + 4 * A[0, 1] * A[1, 0] + A[1, 1] ** 2) / 2 + A[0, 0] / 2 + A[1, 1] / 2)
    raises(ValueError, lambda: JointEigenDistribution(Matrix([[1, 0], [2, 1]])))