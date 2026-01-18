from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.matrices.common import _MinimalMatrix, _CastableMatrix
from sympy.matrices.matrices import MatrixReductions
from sympy.testing.pytest import raises
from sympy.matrices import Matrix, zeros
from sympy.core.symbol import Symbol
from sympy.core.numbers import Rational
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.simplify import simplify
from sympy.abc import x
def test_issue_17827():
    C = Matrix([[3, 4, -1, 1], [9, 12, -3, 3], [0, 2, 1, 3], [2, 3, 0, -2], [0, 3, 3, -5], [8, 15, 0, 6]])
    D = C.elementary_row_op('n<->m', row1=2, row2=5)
    E = C.elementary_row_op('n->n+km', row1=5, row2=3, k=-4)
    F = C.elementary_row_op('n->kn', row=5, k=2)
    assert D[5, :] == Matrix([[0, 2, 1, 3]])
    assert E[5, :] == Matrix([[0, 3, 0, 14]])
    assert F[5, :] == Matrix([[16, 30, 0, 12]])
    raises(ValueError, lambda: C.elementary_row_op('n<->m', row1=2, row2=6))
    raises(ValueError, lambda: C.elementary_row_op('n->kn', row=7, k=2))
    raises(ValueError, lambda: C.elementary_row_op('n->n+km', row1=-1, row2=5, k=2))