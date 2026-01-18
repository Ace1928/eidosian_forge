from sympy.core import I, symbols, Basic, Mul, S
from sympy.core.mul import mul
from sympy.functions import adjoint, transpose
from sympy.matrices.common import ShapeError
from sympy.matrices import (Identity, Inverse, Matrix, MatrixSymbol, ZeroMatrix,
from sympy.matrices.expressions import Adjoint, Transpose, det, MatPow
from sympy.matrices.expressions.special import GenericIdentity
from sympy.matrices.expressions.matmul import (factor_in_front, remove_ids,
from sympy.strategies import null_safe
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.symbol import Symbol
from sympy.testing.pytest import XFAIL, raises
def test_issue_23519():
    N = Symbol('N', integer=True)
    M1 = MatrixSymbol('M1', N, N)
    M2 = MatrixSymbol('M2', N, N)
    I = Identity(N)
    z = M2 + 2 * (M2 + I) * M1 + I
    assert z.coeff(M1) == 2 * I + 2 * M2