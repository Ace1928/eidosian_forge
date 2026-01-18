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
def test_col_op():
    e = eye_Reductions(3)
    raises(ValueError, lambda: e.elementary_col_op('abc'))
    raises(ValueError, lambda: e.elementary_col_op())
    raises(ValueError, lambda: e.elementary_col_op('n->kn', col=5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->kn', col=-5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=1, col2=5))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=5, col2=1))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=-5, col2=1))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=1, col2=-5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=1, col2=5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=5, col2=1, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=-5, col2=1, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=1, col2=-5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=1, col2=1, k=5))
    assert e.elementary_col_op('n->kn', 0, 5) == Matrix([[5, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_col_op('n->kn', 1, 5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_col_op('n->kn', col=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_col_op('n->kn', col1=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_col_op('n<->m', 0, 1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_col_op('n<->m', col1=0, col2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_col_op('n<->m', col=0, col2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_col_op('n->n+km', 0, 5, 1) == Matrix([[1, 0, 0], [5, 1, 0], [0, 0, 1]])
    assert e.elementary_col_op('n->n+km', col=0, k=5, col2=1) == Matrix([[1, 0, 0], [5, 1, 0], [0, 0, 1]])
    assert e.elementary_col_op('n->n+km', col1=0, k=5, col2=1) == Matrix([[1, 0, 0], [5, 1, 0], [0, 0, 1]])
    a = ReductionsOnlyMatrix(2, 3, [0] * 6)
    assert a.elementary_col_op('n->kn', 1, 5) == Matrix(2, 3, [0] * 6)
    assert a.elementary_col_op('n<->m', 0, 1) == Matrix(2, 3, [0] * 6)
    assert a.elementary_col_op('n->n+km', 0, 5, 1) == Matrix(2, 3, [0] * 6)