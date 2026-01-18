from sympy.combinatorics import Permutation
from sympy.core.expr import unchanged
from sympy.matrices import Matrix
from sympy.matrices.expressions import \
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix, Identity
from sympy.matrices.expressions.permutation import \
from sympy.testing.pytest import raises
from sympy.core.symbol import Symbol
def test_MatrixPermute_rewrite_MatMul():
    p = Permutation(0, 1, 2)
    A = MatrixSymbol('A', 3, 3)
    assert MatrixPermute(A, p, 0).rewrite(MatMul).as_explicit() == MatrixPermute(A, p, 0).as_explicit()
    assert MatrixPermute(A, p, 1).rewrite(MatMul).as_explicit() == MatrixPermute(A, p, 1).as_explicit()