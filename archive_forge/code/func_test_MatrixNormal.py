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
def test_MatrixNormal():
    M = MatrixNormal('M', [[5, 6]], [4], [[2, 1], [1, 2]])
    assert M.pspace.distribution.set == MatrixSet(1, 2, S.Reals)
    X = MatrixSymbol('X', 1, 2)
    term1 = exp(-Trace(Matrix([[S(2) / 3, -S(1) / 3], [-S(1) / 3, S(2) / 3]]) * (Matrix([[-5], [-6]]) + X.T) * Matrix([[S(1) / 4]]) * (Matrix([[-5, -6]]) + X)) / 2)
    assert density(M)(X).doit() == sqrt(3) * term1 / (24 * pi)
    assert density(M)([[7, 8]]).doit() == sqrt(3) * exp(-S(1) / 3) / (24 * pi)
    d, n = symbols('d n', positive=True, integer=True)
    SM2 = MatrixSymbol('SM2', d, d)
    SM1 = MatrixSymbol('SM1', n, n)
    LM = MatrixSymbol('LM', n, d)
    Y = MatrixSymbol('Y', n, d)
    M = MatrixNormal('M', LM, SM1, SM2)
    exprd = (2 * pi) ** (-d * n / 2) * exp(-Trace(SM2 ** (-1) * (-LM.T + Y.T) * SM1 ** (-1) * (-LM + Y)) / 2) * Determinant(SM1) ** (-d / 2) * Determinant(SM2) ** (-n / 2)
    assert density(M)(Y).doit() == exprd
    raises(ValueError, lambda: density(M)(1))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [0, 1]], [[1, 0], [2, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [2, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [0, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [2]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [2, 1]], [[1, 0], [0]]))
    raises(ValueError, lambda: MatrixNormal('M', [[1, 2]], [[1, 0], [0, 1]], [[1, 0]]))
    raises(ValueError, lambda: MatrixNormal('M', [[1, 2]], [1], [[1, 0]]))