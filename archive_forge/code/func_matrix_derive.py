import operator
from functools import reduce, singledispatch
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.matrices.expressions.hadamard import HadamardProduct
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import (MatrixExpr, MatrixSymbol)
from sympy.matrices.expressions.special import Identity, OneMatrix
from sympy.matrices.expressions.transpose import Transpose
from sympy.combinatorics.permutations import _af_invert
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.tensor.array.expressions.array_expressions import (
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
def matrix_derive(expr, x):
    from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    ce = convert_matrix_to_array(expr)
    dce = array_derive(ce, x)
    return convert_array_to_matrix(dce).doit()