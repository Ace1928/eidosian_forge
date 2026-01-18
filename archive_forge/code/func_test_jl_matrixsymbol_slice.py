from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_jl_matrixsymbol_slice():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    D = MatrixSymbol('D', 2, 1)
    name_expr = ('test', [Equality(B, A[0, :]), Equality(C, A[1, :]), Equality(D, A[:, 2])])
    result, = codegen(name_expr, 'Julia', header=False, empty=False)
    source = result[1]
    expected = 'function test(A)\n    B = A[1,:]\n    C = A[2,:]\n    D = A[:,3]\n    return B, C, D\nend\n'
    assert source == expected