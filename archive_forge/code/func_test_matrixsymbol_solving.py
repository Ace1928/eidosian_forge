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
def test_matrixsymbol_solving():
    A = MatrixSymbol('A', 2, 2)
    B = MatrixSymbol('B', 2, 2)
    Z = ZeroMatrix(2, 2)
    assert -(-A + B) - A + B == Z
    assert (-(-A + B) - A + B).simplify() == Z
    assert (-(-A + B) - A + B).expand() == Z
    assert (-(-A + B) - A + B - Z).simplify() == Z
    assert (-(-A + B) - A + B - Z).expand() == Z
    assert (A * (A + B) + B * (A.T + B.T)).expand() == A ** 2 + A * B + B * A.T + B * B.T