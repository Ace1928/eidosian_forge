from sympy.combinatorics import Permutation
from sympy.core.expr import unchanged
from sympy.matrices import Matrix
from sympy.matrices.expressions import \
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix, Identity
from sympy.matrices.expressions.permutation import \
from sympy.testing.pytest import raises
from sympy.core.symbol import Symbol
def test_PermutationMatrix_determinant():
    P = PermutationMatrix(Permutation([0, 1, 2]))
    assert Determinant(P).doit() == 1
    P = PermutationMatrix(Permutation([0, 2, 1]))
    assert Determinant(P).doit() == -1
    P = PermutationMatrix(Permutation([2, 0, 1]))
    assert Determinant(P).doit() == 1