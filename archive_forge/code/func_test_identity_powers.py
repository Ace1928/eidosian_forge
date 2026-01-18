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
def test_identity_powers():
    M = Identity(n)
    assert MatPow(M, 3).doit() == M ** 3
    assert M ** n == M
    assert MatPow(M, 0).doit() == M ** 2
    assert M ** (-2) == M
    assert MatPow(M, -2).doit() == M ** 0
    N = Identity(3)
    assert MatPow(N, 2).doit() == N ** n
    assert MatPow(N, 3).doit() == N
    assert MatPow(N, -2).doit() == N ** 4
    assert MatPow(N, 2).doit() == N ** 0