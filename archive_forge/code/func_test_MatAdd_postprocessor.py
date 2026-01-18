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
def test_MatAdd_postprocessor():
    z = zeros(2)
    assert Add(0, z) == Add(z, 0) == z
    a = Add(S.Infinity, z)
    assert a == Add(z, S.Infinity)
    assert isinstance(a, Add)
    assert a.args == (S.Infinity, z)
    a = Add(S.ComplexInfinity, z)
    assert a == Add(z, S.ComplexInfinity)
    assert isinstance(a, Add)
    assert a.args == (S.ComplexInfinity, z)
    a = Add(z, S.NaN)
    assert isinstance(a, Add)
    assert a.args == (S.NaN, z)
    M = Matrix([[1, 2], [3, 4]])
    a = Add(x, M)
    assert a == Add(M, x)
    assert isinstance(a, Add)
    assert a.args == (x, M)
    A = MatrixSymbol('A', 2, 2)
    assert Add(A, M) == Add(M, A) == A + M
    a = Add(x, M, A)
    assert a == Add(M, x, A) == Add(M, A, x) == Add(x, A, M) == Add(A, x, M) == Add(A, M, x)
    assert isinstance(a, Add)
    assert a.args == (x, A + M)
    assert Add(M, M) == 2 * M
    assert Add(M, A, M) == Add(M, M, A) == Add(A, M, M) == A + 2 * M
    a = Add(A, x, M, M, x)
    assert isinstance(a, Add)
    assert a.args == (2 * x, A + 2 * M)