from sympy.combinatorics import Permutation
from sympy.core.expr import unchanged
from sympy.matrices import Matrix
from sympy.matrices.expressions import \
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix, Identity
from sympy.matrices.expressions.permutation import \
from sympy.testing.pytest import raises
from sympy.core.symbol import Symbol
def test_MatrixPermute_doit():
    p = Permutation(0, 1, 2)
    A = MatrixSymbol('A', 3, 3)
    assert MatrixPermute(A, p).doit() == MatrixPermute(A, p)
    p = Permutation(0, size=3)
    A = MatrixSymbol('A', 3, 3)
    assert MatrixPermute(A, p).doit().as_explicit() == MatrixPermute(A, p).as_explicit()
    p = Permutation(0, 1, 2)
    A = Identity(3)
    assert MatrixPermute(A, p, 0).doit().as_explicit() == MatrixPermute(A, p, 0).as_explicit()
    assert MatrixPermute(A, p, 1).doit().as_explicit() == MatrixPermute(A, p, 1).as_explicit()
    A = ZeroMatrix(3, 3)
    assert MatrixPermute(A, p).doit() == A
    A = OneMatrix(3, 3)
    assert MatrixPermute(A, p).doit() == A
    A = MatrixSymbol('A', 4, 4)
    p1 = Permutation(0, 1, 2, 3)
    p2 = Permutation(0, 2, 3, 1)
    expr = MatrixPermute(MatrixPermute(A, p1, 0), p2, 0)
    assert expr.as_explicit() == expr.doit().as_explicit()
    expr = MatrixPermute(MatrixPermute(A, p1, 1), p2, 1)
    assert expr.as_explicit() == expr.doit().as_explicit()