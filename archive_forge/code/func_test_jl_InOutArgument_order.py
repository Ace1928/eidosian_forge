from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_jl_InOutArgument_order():
    expr = Equality(x, x ** 2 + y)
    name_expr = ('test', expr)
    result, = codegen(name_expr, 'Julia', header=False, empty=False, argument_sequence=(x, y))
    source = result[1]
    expected = 'function test(x, y)\n    x = x .^ 2 + y\n    return x\nend\n'
    assert source == expected
    expr = Equality(x, x ** 2 + y)
    name_expr = ('test', expr)
    result, = codegen(name_expr, 'Julia', header=False, empty=False)
    source = result[1]
    expected = 'function test(x, y)\n    x = x .^ 2 + y\n    return x\nend\n'
    assert source == expected