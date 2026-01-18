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
def test_Wishart():
    W = Wishart('W', 5, [[1, 0], [0, 1]])
    assert W.pspace.distribution.set == MatrixSet(2, 2, S.Reals)
    X = MatrixSymbol('X', 2, 2)
    term1 = exp(Trace(Matrix([[-S(1) / 2, 0], [0, -S(1) / 2]]) * X))
    assert density(W)(X).doit() == term1 * Determinant(X) / (24 * pi)
    assert density(W)([[2, 1], [1, 2]]).doit() == exp(-2) / (8 * pi)
    n = symbols('n', positive=True)
    d = symbols('d', positive=True, integer=True)
    Y = MatrixSymbol('Y', d, d)
    SM = MatrixSymbol('SM', d, d)
    W = Wishart('W', n, SM)
    k = Dummy('k')
    exprd = 2 ** (-d * n / 2) * pi ** (-d * (d - 1) / 4) * exp(Trace(-(S(1) / 2) * SM ** (-1) * Y)) * Determinant(SM) ** (-n / 2) * Determinant(Y) ** (-d / 2 + n / 2 - S(1) / 2) / Product(gamma(-k / 2 + n / 2 + S(1) / 2), (k, 1, d))
    assert density(W)(Y).dummy_eq(exprd)
    raises(ValueError, lambda: density(W)(1))
    raises(ValueError, lambda: Wishart('W', -1, [[1, 0], [0, 1]]))
    raises(ValueError, lambda: Wishart('W', -1, [[1, 0], [2, 1]]))
    raises(ValueError, lambda: Wishart('W', 2, [[1, 0], [0]]))