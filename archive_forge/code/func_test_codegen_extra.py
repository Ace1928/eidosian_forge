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
def test_codegen_extra():
    if not np:
        skip('NumPy not installed')
    M = MatrixSymbol('M', 2, 2)
    N = MatrixSymbol('N', 2, 2)
    P = MatrixSymbol('P', 2, 2)
    Q = MatrixSymbol('Q', 2, 2)
    ma = np.array([[1, 2], [3, 4]])
    mb = np.array([[1, -2], [-1, 3]])
    mc = np.array([[2, 0], [1, 2]])
    md = np.array([[1, -1], [4, 7]])
    cg = ArrayTensorProduct(M, N)
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == np.einsum(ma, [0, 1], mb, [2, 3])).all()
    cg = ArrayAdd(M, N)
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == ma + mb).all()
    cg = ArrayAdd(M, N, P)
    f = lambdify((M, N, P), cg, 'numpy')
    assert (f(ma, mb, mc) == ma + mb + mc).all()
    cg = ArrayAdd(M, N, P, Q)
    f = lambdify((M, N, P, Q), cg, 'numpy')
    assert (f(ma, mb, mc, md) == ma + mb + mc + md).all()
    cg = PermuteDims(M, [1, 0])
    f = lambdify((M,), cg, 'numpy')
    assert (f(ma) == ma.T).all()
    cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == np.transpose(np.einsum(ma, [0, 1], mb, [2, 3]), (1, 2, 3, 0))).all()
    cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == np.diagonal(np.einsum(ma, [0, 1], mb, [2, 3]), axis1=1, axis2=2)).all()