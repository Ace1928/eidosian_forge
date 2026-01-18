from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
@XFAIL
def test_m_numbersymbol_no_inline():
    name_expr = ('test', [pi ** Catalan, EulerGamma])
    result, = codegen(name_expr, 'Octave', header=False, empty=False, inline=False)
    source = result[1]
    expected = 'function [out1, out2] = test()\n  Catalan = 0.915965594177219;  % constant\n  EulerGamma = 0.5772156649015329;  % constant\n  out1 = pi^Catalan;\n  out2 = EulerGamma;\nend\n'
    assert source == expected