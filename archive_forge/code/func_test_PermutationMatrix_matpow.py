from sympy.combinatorics import Permutation
from sympy.core.expr import unchanged
from sympy.matrices import Matrix
from sympy.matrices.expressions import \
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix, Identity
from sympy.matrices.expressions.permutation import \
from sympy.testing.pytest import raises
from sympy.core.symbol import Symbol
def test_PermutationMatrix_matpow():
    p1 = Permutation([1, 2, 0])
    P1 = PermutationMatrix(p1)
    p2 = Permutation([2, 0, 1])
    P2 = PermutationMatrix(p2)
    assert P1 ** 2 == P2
    assert P1 ** 3 == Identity(3)