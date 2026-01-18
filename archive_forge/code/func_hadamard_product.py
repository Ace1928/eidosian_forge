from collections import Counter
from sympy.core import Mul, sympify
from sympy.core.add import Add
from sympy.core.expr import ExprBuilder
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import log
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix
from sympy.strategies import (
from sympy.utilities.exceptions import sympy_deprecation_warning
def hadamard_product(*matrices):
    """
    Return the elementwise (aka Hadamard) product of matrices.

    Examples
    ========

    >>> from sympy import hadamard_product, MatrixSymbol
    >>> A = MatrixSymbol('A', 2, 3)
    >>> B = MatrixSymbol('B', 2, 3)
    >>> hadamard_product(A)
    A
    >>> hadamard_product(A, B)
    HadamardProduct(A, B)
    >>> hadamard_product(A, B)[0, 1]
    A[0, 1]*B[0, 1]
    """
    if not matrices:
        raise TypeError('Empty Hadamard product is undefined')
    if len(matrices) == 1:
        return matrices[0]
    return HadamardProduct(*matrices).doit()