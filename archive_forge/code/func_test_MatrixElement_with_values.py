from sympy.concrete.summations import Sum
from sympy.core.exprtools import gcd_terms
from sympy.core.function import (diff, expand)
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol, Str)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.polys.polytools import factor
from sympy.core import (S, symbols, Add, Mul, SympifyError, Rational,
from sympy.functions import sin, cos, tan, sqrt, cbrt, exp
from sympy.simplify import simplify
from sympy.matrices import (ImmutableMatrix, Inverse, MatAdd, MatMul,
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions.determinant import Determinant, det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.testing.pytest import raises, XFAIL
def test_MatrixElement_with_values():
    x, y, z, w = symbols('x y z w')
    M = Matrix([[x, y], [z, w]])
    i, j = symbols('i, j')
    Mij = M[i, j]
    assert isinstance(Mij, MatrixElement)
    Ms = SparseMatrix([[2, 3], [4, 5]])
    msij = Ms[i, j]
    assert isinstance(msij, MatrixElement)
    for oi, oj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert Mij.subs({i: oi, j: oj}) == M[oi, oj]
        assert msij.subs({i: oi, j: oj}) == Ms[oi, oj]
    A = MatrixSymbol('A', 2, 2)
    assert A[0, 0].subs(A, M) == x
    assert A[i, j].subs(A, M) == M[i, j]
    assert M[i, j].subs(M, A) == A[i, j]
    assert isinstance(M[3 * i - 2, j], MatrixElement)
    assert M[3 * i - 2, j].subs({i: 1, j: 0}) == M[1, 0]
    assert isinstance(M[i, 0], MatrixElement)
    assert M[i, 0].subs(i, 0) == M[0, 0]
    assert M[0, i].subs(i, 1) == M[0, 1]
    assert M[i, j].diff(x) == Matrix([[1, 0], [0, 0]])[i, j]
    raises(ValueError, lambda: M[i, 2])
    raises(ValueError, lambda: M[i, -1])
    raises(ValueError, lambda: M[2, i])
    raises(ValueError, lambda: M[-1, i])