from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_jl_not_supported():
    f = Function('f')
    name_expr = ('test', [f(x).diff(x), S.ComplexInfinity])
    result, = codegen(name_expr, 'Julia', header=False, empty=False)
    source = result[1]
    expected = 'function test(x)\n    # unsupported: Derivative(f(x), x)\n    # unsupported: zoo\n    out1 = Derivative(f(x), x)\n    out2 = zoo\n    return out1, out2\nend\n'
    assert source == expected