from sympy.concrete.summations import Sum
from sympy.core.mod import Mod
from sympy.core.relational import (Equality, Unequality)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
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
from sympy.printing.numpy import JaxPrinter, _jax_known_constants, _jax_known_functions
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.testing.pytest import skip, raises
from sympy.external import import_module
def test_jax_matsolve():
    if not jax:
        skip('JAX not installed')
    M = MatrixSymbol('M', 3, 3)
    x = MatrixSymbol('x', 3, 1)
    expr = M ** (-1) * x + x
    matsolve_expr = MatrixSolve(M, x) + x
    f = lambdify((M, x), expr, 'jax')
    f_matsolve = lambdify((M, x), matsolve_expr, 'jax')
    m0 = jax.numpy.array([[1, 2, 3], [3, 2, 5], [5, 6, 7]])
    assert jax.numpy.linalg.matrix_rank(m0) == 3
    x0 = jax.numpy.array([3, 4, 5])
    assert jax.numpy.allclose(f_matsolve(m0, x0), f(m0, x0))