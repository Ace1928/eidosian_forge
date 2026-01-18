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
def test_MatrixStudentT():
    M = MatrixStudentT('M', 2, [[5, 6]], [[2, 1], [1, 2]], [4])
    assert M.pspace.distribution.set == MatrixSet(1, 2, S.Reals)
    X = MatrixSymbol('X', 1, 2)
    D = pi ** (-1.0) * Determinant(Matrix([[4]])) ** (-1.0) * Determinant(Matrix([[2, 1], [1, 2]])) ** (-0.5) / Determinant(Matrix([[S(1) / 4]]) * (Matrix([[-5, -6]]) + X) * Matrix([[S(2) / 3, -S(1) / 3], [-S(1) / 3, S(2) / 3]]) * (Matrix([[-5], [-6]]) + X.T) + Matrix([[1]])) ** 2
    assert density(M)(X) == D
    v = symbols('v', positive=True)
    n, p = (1, 2)
    Omega = MatrixSymbol('Omega', p, p)
    Sigma = MatrixSymbol('Sigma', n, n)
    Location = MatrixSymbol('Location', n, p)
    Y = MatrixSymbol('Y', n, p)
    M = MatrixStudentT('M', v, Location, Omega, Sigma)
    exprd = gamma(v / 2 + 1) * Determinant(Matrix([[1]]) + Sigma ** (-1) * (-Location + Y) * Omega ** (-1) * (-Location.T + Y.T)) ** (-v / 2 - 1) / (pi * gamma(v / 2) * sqrt(Determinant(Omega)) * Determinant(Sigma))
    assert density(M)(Y) == exprd
    raises(ValueError, lambda: density(M)(1))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [0, 1]], [[1, 0], [2, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [2, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [0, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [2]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [2, 1]], [[1], [2]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [[1, 2]], [[1, 0], [0, 1]], [[1, 0]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [[1, 2]], [1], [[1, 0]]))
    raises(ValueError, lambda: MatrixStudentT('M', -1, [1, 2], [[1, 0], [0, 1]], [4]))