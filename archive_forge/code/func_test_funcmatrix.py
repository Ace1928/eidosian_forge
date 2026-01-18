from sympy.core import symbols, Lambda
from sympy.functions import KroneckerDelta
from sympy.matrices import Matrix
from sympy.matrices.expressions import FunctionMatrix, MatrixExpr, Identity
from sympy.testing.pytest import raises, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
def test_funcmatrix():
    i, j = symbols('i,j')
    X = FunctionMatrix(3, 3, Lambda((i, j), i - j))
    assert X[1, 1] == 0
    assert X[1, 2] == -1
    assert X.shape == (3, 3)
    assert X.rows == X.cols == 3
    assert Matrix(X) == Matrix(3, 3, lambda i, j: i - j)
    assert isinstance(X * X + X, MatrixExpr)