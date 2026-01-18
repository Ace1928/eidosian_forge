from mpmath.matrices.matrices import _matrix
from sympy.core import Basic, Dict, Tuple
from sympy.core.numbers import Integer
from sympy.core.cache import cacheit
from sympy.core.sympify import _sympy_converter as sympify_converter, _sympify
from sympy.matrices.dense import DenseMatrix
from sympy.matrices.expressions import MatrixExpr
from sympy.matrices.matrices import MatrixBase
from sympy.matrices.repmatrix import RepMatrix
from sympy.matrices.sparse import SparseRepMatrix
from sympy.multipledispatch import dispatch
def sympify_matrix(arg):
    return arg.as_immutable()