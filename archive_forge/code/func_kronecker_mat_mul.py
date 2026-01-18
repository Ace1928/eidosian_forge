from functools import reduce
from math import prod
from sympy.core import Mul, sympify
from sympy.functions import adjoint
from sympy.matrices.common import ShapeError
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.transpose import transpose
from sympy.matrices.expressions.special import Identity
from sympy.matrices.matrices import MatrixBase
from sympy.strategies import (
from sympy.strategies.traverse import bottom_up
from sympy.utilities import sift
from .matadd import MatAdd
from .matmul import MatMul
from .matpow import MatPow
def kronecker_mat_mul(expr):
    factor, matrices = expr.as_coeff_matrices()
    i = 0
    while i < len(matrices) - 1:
        A, B = matrices[i:i + 2]
        if isinstance(A, KroneckerProduct) and isinstance(B, KroneckerProduct):
            matrices[i] = A._kronecker_mul(B)
            matrices.pop(i + 1)
        else:
            i += 1
    return factor * MatMul(*matrices)