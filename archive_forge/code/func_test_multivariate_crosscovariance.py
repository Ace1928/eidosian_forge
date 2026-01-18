from sympy.stats import Expectation, Normal, Variance, Covariance
from sympy.testing.pytest import raises
from sympy.core.symbol import symbols
from sympy.matrices.common import ShapeError
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.stats.rv import RandomMatrixSymbol
from sympy.stats.symbolic_multivariate_probability import (ExpectationMatrix,
def test_multivariate_crosscovariance():
    raises(ShapeError, lambda: Covariance(X, Y.T))
    raises(ShapeError, lambda: Covariance(X, A))
    expr = Covariance(a.T, b.T)
    assert expr.shape == (1, 1)
    assert expr.expand() == ZeroMatrix(1, 1)
    expr = Covariance(a, b)
    assert expr == Covariance(a, b) == CrossCovarianceMatrix(a, b)
    assert expr.expand() == ZeroMatrix(k, k)
    assert expr.shape == (k, k)
    assert expr.rows == k
    assert expr.cols == k
    assert isinstance(expr, CrossCovarianceMatrix)
    expr = Covariance(A * X + a, b)
    assert expr.expand() == ZeroMatrix(k, k)
    expr = Covariance(X, Y)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == expr
    expr = Covariance(X, X)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == VarianceMatrix(X)
    expr = Covariance(X + Y, Z)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == CrossCovarianceMatrix(X, Z) + CrossCovarianceMatrix(Y, Z)
    expr = Covariance(A * X, Y)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == A * CrossCovarianceMatrix(X, Y)
    expr = Covariance(X, B * Y)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == CrossCovarianceMatrix(X, Y) * B.T
    expr = Covariance(A * X + a, B.T * Y + b)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == A * CrossCovarianceMatrix(X, Y) * B
    expr = Covariance(A * X + B * Y + a, C.T * Z + D.T * W + b)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == A * CrossCovarianceMatrix(X, W) * D + A * CrossCovarianceMatrix(X, Z) * C + B * CrossCovarianceMatrix(Y, W) * D + B * CrossCovarianceMatrix(Y, Z) * C