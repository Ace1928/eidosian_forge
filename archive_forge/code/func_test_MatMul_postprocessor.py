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
def test_MatMul_postprocessor():
    z = zeros(2)
    z1 = ZeroMatrix(2, 2)
    assert Mul(0, z) == Mul(z, 0) in [z, z1]
    M = Matrix([[1, 2], [3, 4]])
    Mx = Matrix([[x, 2 * x], [3 * x, 4 * x]])
    assert Mul(x, M) == Mul(M, x) == Mx
    A = MatrixSymbol('A', 2, 2)
    assert Mul(A, M) == MatMul(A, M)
    assert Mul(M, A) == MatMul(M, A)
    a = Mul(x, M, A)
    b = Mul(M, x, A)
    c = Mul(M, A, x)
    assert a == b == c == MatMul(Mx, A)
    a = Mul(x, A, M)
    b = Mul(A, x, M)
    c = Mul(A, M, x)
    assert a == b == c == MatMul(A, Mx)
    assert Mul(M, M) == M ** 2
    assert Mul(A, M, M) == MatMul(A, M ** 2)
    assert Mul(M, M, A) == MatMul(M ** 2, A)
    assert Mul(M, A, M) == MatMul(M, A, M)
    assert Mul(A, x, M, M, x) == MatMul(A, Mx ** 2)