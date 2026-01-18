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
@XFAIL
def test_matmul_args_cnc_symbols():
    a, b = symbols('a b', commutative=False)
    assert MatMul(n, a, b, A, A.T).args_cnc() == [[n], [a, b, A, A.T]]
    assert MatMul(n, a, A, b, A.T).args_cnc() == [[n], [a, A, b, A.T]]