from sympy.core.basic import Basic
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.matrices.matrices import MatrixBase
from sympy.matrices.common import NonSquareMatrixError
def get_arg_key(x):
    a = trace_arg.args[x]
    if isinstance(a, Transpose):
        a = a.arg
    return default_sort_key(a)