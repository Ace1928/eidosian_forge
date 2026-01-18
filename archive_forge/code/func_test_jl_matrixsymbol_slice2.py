from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_jl_matrixsymbol_slice2():
    A = MatrixSymbol('A', 3, 4)
    B = MatrixSymbol('B', 2, 2)
    C = MatrixSymbol('C', 2, 2)
    name_expr = ('test', [Equality(B, A[0:2, 0:2]), Equality(C, A[0:2, 1:3])])
    result, = codegen(name_expr, 'Julia', header=False, empty=False)
    source = result[1]
    expected = 'function test(A)\n    B = A[1:2,1:2]\n    C = A[1:2,2:3]\n    return B, C\nend\n'
    assert source == expected