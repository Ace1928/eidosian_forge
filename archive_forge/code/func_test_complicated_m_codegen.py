from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_complicated_m_codegen():
    from sympy.functions.elementary.trigonometric import cos, sin, tan
    name_expr = ('testlong', [((sin(x) + cos(y) + tan(z)) ** 3).expand(), cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))])
    result = codegen(name_expr, 'Octave', header=False, empty=False)
    assert result[0][0] == 'testlong.m'
    source = result[0][1]
    expected = 'function [out1, out2] = testlong(x, y, z)\n  out1 = sin(x).^3 + 3*sin(x).^2.*cos(y) + 3*sin(x).^2.*tan(z) + 3*sin(x).*cos(y).^2 + 6*sin(x).*cos(y).*tan(z) + 3*sin(x).*tan(z).^2 + cos(y).^3 + 3*cos(y).^2.*tan(z) + 3*cos(y).*tan(z).^2 + tan(z).^3;\n  out2 = cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))));\nend\n'
    assert source == expected