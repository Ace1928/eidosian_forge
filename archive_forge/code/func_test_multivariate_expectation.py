from sympy.stats import Expectation, Normal, Variance, Covariance
from sympy.testing.pytest import raises
from sympy.core.symbol import symbols
from sympy.matrices.common import ShapeError
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.stats.rv import RandomMatrixSymbol
from sympy.stats.symbolic_multivariate_probability import (ExpectationMatrix,
def test_multivariate_expectation():
    expr = Expectation(a)
    assert expr == Expectation(a) == ExpectationMatrix(a)
    assert expr.expand() == a
    expr = Expectation(X)
    assert expr == Expectation(X) == ExpectationMatrix(X)
    assert expr.shape == (k, 1)
    assert expr.rows == k
    assert expr.cols == 1
    assert isinstance(expr, ExpectationMatrix)
    expr = Expectation(A * X + b)
    assert expr == ExpectationMatrix(A * X + b)
    assert expr.expand() == A * ExpectationMatrix(X) + b
    assert isinstance(expr, ExpectationMatrix)
    assert expr.shape == (k, 1)
    expr = Expectation(m1 * X2)
    assert expr.expand() == expr
    expr = Expectation(A2 * m1 * B2 * X2)
    assert expr.args[0].args == (A2, m1, B2, X2)
    assert expr.expand() == A2 * ExpectationMatrix(m1 * B2 * X2)
    expr = Expectation((X + Y) * (X - Y).T)
    assert expr.expand() == ExpectationMatrix(X * X.T) - ExpectationMatrix(X * Y.T) + ExpectationMatrix(Y * X.T) - ExpectationMatrix(Y * Y.T)
    expr = Expectation(A * X + B * Y)
    assert expr.expand() == A * ExpectationMatrix(X) + B * ExpectationMatrix(Y)
    assert Expectation(m1).doit() == Matrix([[1, 2 * j], [0, 0]])
    x1 = Matrix([[Normal('N11', 11, 1), Normal('N12', 12, 1)], [Normal('N21', 21, 1), Normal('N22', 22, 1)]])
    x2 = Matrix([[Normal('M11', 1, 1), Normal('M12', 2, 1)], [Normal('M21', 3, 1), Normal('M22', 4, 1)]])
    assert Expectation(Expectation(x1 + x2)).doit(deep=False) == ExpectationMatrix(x1 + x2)
    assert Expectation(Expectation(x1 + x2)).doit() == Matrix([[12, 14], [24, 26]])