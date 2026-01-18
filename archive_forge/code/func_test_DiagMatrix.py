from sympy.matrices.expressions import MatrixSymbol
from sympy.matrices.expressions.diagonal import DiagonalMatrix, DiagonalOf, DiagMatrix, diagonalize_vector
from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.special import Identity
from sympy.testing.pytest import raises
def test_DiagMatrix():
    x = MatrixSymbol('x', n, 1)
    d = DiagMatrix(x)
    assert d.shape == (n, n)
    assert d[0, 1] == 0
    assert d[0, 0] == x[0, 0]
    a = MatrixSymbol('a', 1, 1)
    d = diagonalize_vector(a)
    assert isinstance(d, MatrixSymbol)
    assert a == d
    assert diagonalize_vector(Identity(3)) == Identity(3)
    assert DiagMatrix(Identity(3)).doit() == Identity(3)
    assert isinstance(DiagMatrix(Identity(3)), DiagMatrix)
    assert DiagMatrix(x).T == DiagMatrix(x)
    assert diagonalize_vector(x.T) == DiagMatrix(x)
    dx = DiagMatrix(x)
    assert dx[0, 0] == x[0, 0]
    assert dx[1, 1] == x[1, 0]
    assert dx[0, 1] == 0
    assert dx[0, m] == x[0, 0] * KroneckerDelta(0, m)
    z = MatrixSymbol('z', 1, n)
    dz = DiagMatrix(z)
    assert dz[0, 0] == z[0, 0]
    assert dz[1, 1] == z[0, 1]
    assert dz[0, 1] == 0
    assert dz[0, m] == z[0, m] * KroneckerDelta(0, m)
    v = MatrixSymbol('v', 3, 1)
    dv = DiagMatrix(v)
    assert dv.as_explicit() == Matrix([[v[0, 0], 0, 0], [0, v[1, 0], 0], [0, 0, v[2, 0]]])
    v = MatrixSymbol('v', 1, 3)
    dv = DiagMatrix(v)
    assert dv.as_explicit() == Matrix([[v[0, 0], 0, 0], [0, v[0, 1], 0], [0, 0, v[0, 2]]])
    dv = DiagMatrix(3 * v)
    assert dv.args == (3 * v,)
    assert dv.doit() == 3 * DiagMatrix(v)
    assert isinstance(dv.doit(), MatMul)
    a = MatrixSymbol('a', 3, 1).as_explicit()
    expr = DiagMatrix(a)
    result = Matrix([[a[0, 0], 0, 0], [0, a[1, 0], 0], [0, 0, a[2, 0]]])
    assert expr.doit() == result
    expr = DiagMatrix(a.T)
    assert expr.doit() == result