from sympy.concrete.products import Product
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices import Determinant, Matrix, Trace, MatrixSymbol, MatrixSet
from sympy.stats import density, sample
from sympy.stats.matrix_distributions import (MatrixGammaDistribution,
from sympy.testing.pytest import raises, skip
from sympy.external import import_module
def test_MatrixPSpace():
    M = MatrixGammaDistribution(1, 2, [[2, 1], [1, 2]])
    MP = MatrixPSpace('M', M, 2, 2)
    assert MP.distribution == M
    raises(ValueError, lambda: MatrixPSpace('M', M, 1.2, 2))