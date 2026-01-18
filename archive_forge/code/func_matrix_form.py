from __future__ import annotations
from functools import wraps
from sympy.core import S, Integer, Basic, Mul, Add
from sympy.core.assumptions import check_assumptions
from sympy.core.decorators import call_highest_priority
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.logic import FuzzyBool
from sympy.core.symbol import Str, Dummy, symbols, Symbol
from sympy.core.sympify import SympifyError, _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.matrices import MatrixKind, MatrixBase
from sympy.multipledispatch import dispatch
from sympy.utilities.misc import filldedent
from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
from .special import ZeroMatrix, Identity
from .determinant import Determinant
def matrix_form(self):
    if self.first != 1 and self.higher != 1:
        raise ValueError('higher dimensional array cannot be represented')

    def _get_shape(elem):
        if isinstance(elem, MatrixExpr):
            return elem.shape
        return (None, None)
    if _get_shape(self.first)[1] != _get_shape(self.second)[1]:
        if _get_shape(self.second) == (1, 1):
            return self.first * self.second[0, 0]
        if _get_shape(self.first) == (1, 1):
            return self.first[1, 1] * self.second.T
        raise ValueError('incompatible shapes')
    if self.first != 1:
        return self.first * self.second.T
    else:
        return self.higher