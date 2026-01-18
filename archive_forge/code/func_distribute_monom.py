from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
from sympy.core import Basic, sympify, S
from sympy.core.mul import mul, Mul
from sympy.core.numbers import Number, Integer
from sympy.core.symbol import Dummy
from sympy.functions import adjoint
from sympy.strategies import (rm_id, unpack, typed, flatten, exhaust,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices.matrices import MatrixBase
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.matrices.expressions._shape import validate_matmul_integer as validate
from .inverse import Inverse
from .matexpr import MatrixExpr
from .matpow import MatPow
from .transpose import transpose
from .permutation import PermutationMatrix
from .special import ZeroMatrix, Identity, GenericIdentity, OneMatrix
def distribute_monom(mul):
    """
    Simplify MatMul expressions but distributing
    rational term to MatMul.

    e.g. 2*(A+B) -> 2*A + 2*B
    """
    args = mul.args
    if len(args) == 2:
        from .matadd import MatAdd
        if args[0].is_MatAdd and args[1].is_Rational:
            return MatAdd(*[MatMul(mat, args[1]).doit() for mat in args[0].args])
        if args[1].is_MatAdd and args[0].is_Rational:
            return MatAdd(*[MatMul(args[0], mat).doit() for mat in args[1].args])
    return mul