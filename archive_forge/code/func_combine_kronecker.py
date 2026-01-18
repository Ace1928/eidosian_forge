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
def combine_kronecker(expr):
    """Combine KronekeckerProduct with expression.

    If possible write operations on KroneckerProducts of compatible shapes
    as a single KroneckerProduct.

    Examples
    ========

    >>> from sympy.matrices.expressions import combine_kronecker
    >>> from sympy import MatrixSymbol, KroneckerProduct, symbols
    >>> m, n = symbols(r'm, n', integer=True)
    >>> A = MatrixSymbol('A', m, n)
    >>> B = MatrixSymbol('B', n, m)
    >>> combine_kronecker(KroneckerProduct(A, B)*KroneckerProduct(B, A))
    KroneckerProduct(A*B, B*A)
    >>> combine_kronecker(KroneckerProduct(A, B)+KroneckerProduct(B.T, A.T))
    KroneckerProduct(A + B.T, B + A.T)
    >>> C = MatrixSymbol('C', n, n)
    >>> D = MatrixSymbol('D', m, m)
    >>> combine_kronecker(KroneckerProduct(C, D)**m)
    KroneckerProduct(C**m, D**m)
    """

    def haskron(expr):
        return isinstance(expr, MatrixExpr) and expr.has(KroneckerProduct)
    rule = exhaust(bottom_up(exhaust(condition(haskron, typed({MatAdd: kronecker_mat_add, MatMul: kronecker_mat_mul, MatPow: kronecker_mat_pow})))))
    result = rule(expr)
    doit = getattr(result, 'doit', None)
    if doit is not None:
        return doit()
    else:
        return result