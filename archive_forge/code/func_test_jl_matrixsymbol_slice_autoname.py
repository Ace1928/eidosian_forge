from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_jl_matrixsymbol_slice_autoname():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 1, 3)
    name_expr = ('test', [Equality(B, A[0, :]), A[1, :], A[:, 0], A[:, 1]])
    result, = codegen(name_expr, 'Julia', header=False, empty=False)
    source = result[1]
    expected = 'function test(A)\n    B = A[1,:]\n    out2 = A[2,:]\n    out3 = A[:,1]\n    out4 = A[:,2]\n    return B, out2, out3, out4\nend\n'
    assert source == expected