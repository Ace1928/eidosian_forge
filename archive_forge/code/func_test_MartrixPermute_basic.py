from sympy.combinatorics import Permutation
from sympy.core.expr import unchanged
from sympy.matrices import Matrix
from sympy.matrices.expressions import \
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix, Identity
from sympy.matrices.expressions.permutation import \
from sympy.testing.pytest import raises
from sympy.core.symbol import Symbol
def test_MartrixPermute_basic():
    p = Permutation(0, 1)
    P = PermutationMatrix(p)
    A = MatrixSymbol('A', 2, 2)
    raises(ValueError, lambda: MatrixPermute(Symbol('x'), p))
    raises(ValueError, lambda: MatrixPermute(A, Symbol('x')))
    assert MatrixPermute(A, P) == MatrixPermute(A, p)
    raises(ValueError, lambda: MatrixPermute(A, p, 2))
    pp = Permutation(0, 1, size=3)
    assert MatrixPermute(A, pp) == MatrixPermute(A, p)
    pp = Permutation(0, 1, 2)
    raises(ValueError, lambda: MatrixPermute(A, pp))