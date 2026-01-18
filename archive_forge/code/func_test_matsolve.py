from sympy.concrete.summations import Sum
from sympy.core.mod import Mod
from sympy.core.relational import (Equality, Unequality)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import polygamma
from sympy.functions.special.error_functions import (Si, Ci)
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.utilities.lambdify import lambdify
from sympy.abc import x, i, j, a, b, c, d
from sympy.core import Pow
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.cfunctions import log1p, expm1, hypot, log10, exp2, log2, Sqrt
from sympy.tensor.array import Array
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
from sympy.printing.numpy import NumPyPrinter, SciPyPrinter, _numpy_known_constants, \
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.testing.pytest import skip, raises
from sympy.external import import_module
def test_matsolve():
    if not np:
        skip('NumPy not installed')
    M = MatrixSymbol('M', 3, 3)
    x = MatrixSymbol('x', 3, 1)
    expr = M ** (-1) * x + x
    matsolve_expr = MatrixSolve(M, x) + x
    f = lambdify((M, x), expr)
    f_matsolve = lambdify((M, x), matsolve_expr)
    m0 = np.array([[1, 2, 3], [3, 2, 5], [5, 6, 7]])
    assert np.linalg.matrix_rank(m0) == 3
    x0 = np.array([3, 4, 5])
    assert np.allclose(f_matsolve(m0, x0), f(m0, x0))