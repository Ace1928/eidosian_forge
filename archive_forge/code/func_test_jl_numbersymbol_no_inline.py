from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
@XFAIL
def test_jl_numbersymbol_no_inline():
    name_expr = ('test', [pi ** Catalan, EulerGamma])
    result, = codegen(name_expr, 'Julia', header=False, empty=False, inline=False)
    source = result[1]
    expected = 'function test()\n    Catalan = 0.915965594177219\n    EulerGamma = 0.5772156649015329\n    out1 = pi ^ Catalan\n    out2 = EulerGamma\n    return out1, out2\nend\n'
    assert source == expected