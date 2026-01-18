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
def combine_one_matrices(mul):
    """
    Combine products of OneMatrix

    e.g. OneMatrix(2, 3) * OneMatrix(3, 4) -> 3 * OneMatrix(2, 4)
    """
    factor, args = mul.as_coeff_matrices()
    new_args = [args[0]]
    for B in args[1:]:
        A = new_args[-1]
        if not isinstance(A, OneMatrix) or not isinstance(B, OneMatrix):
            new_args.append(B)
            continue
        new_args.pop()
        new_args.append(OneMatrix(A.shape[0], B.shape[1]))
        factor *= A.shape[1]
    return newmul(factor, *new_args)