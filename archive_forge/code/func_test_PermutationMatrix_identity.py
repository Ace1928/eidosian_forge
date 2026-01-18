from sympy.combinatorics import Permutation
from sympy.core.expr import unchanged
from sympy.matrices import Matrix
from sympy.matrices.expressions import \
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix, Identity
from sympy.matrices.expressions.permutation import \
from sympy.testing.pytest import raises
from sympy.core.symbol import Symbol
def test_PermutationMatrix_identity():
    p = Permutation([0, 1])
    assert PermutationMatrix(p).is_Identity
    p = Permutation([1, 0])
    assert not PermutationMatrix(p).is_Identity