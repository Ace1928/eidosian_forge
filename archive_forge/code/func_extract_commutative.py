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
def extract_commutative(kron):
    c_part = []
    nc_part = []
    for arg in kron.args:
        c, nc = arg.args_cnc()
        c_part.extend(c)
        nc_part.append(Mul._from_args(nc))
    c_part = Mul(*c_part)
    if c_part != 1:
        return c_part * KroneckerProduct(*nc_part)
    return kron