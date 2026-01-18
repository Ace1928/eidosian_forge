from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_m_multifcns_per_file():
    name_expr = [('foo', [2 * x, 3 * y]), ('bar', [y ** 2, 4 * y])]
    result = codegen(name_expr, 'Octave', header=False, empty=False)
    assert result[0][0] == 'foo.m'
    source = result[0][1]
    expected = 'function [out1, out2] = foo(x, y)\n  out1 = 2*x;\n  out2 = 3*y;\nend\nfunction [out1, out2] = bar(y)\n  out1 = y.^2;\n  out2 = 4*y;\nend\n'
    assert source == expected