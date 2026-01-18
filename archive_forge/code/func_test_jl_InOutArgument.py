from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_jl_InOutArgument():
    expr = Equality(x, x ** 2)
    name_expr = ('mysqr', expr)
    result, = codegen(name_expr, 'Julia', header=False, empty=False)
    source = result[1]
    expected = 'function mysqr(x)\n    x = x .^ 2\n    return x\nend\n'
    assert source == expected