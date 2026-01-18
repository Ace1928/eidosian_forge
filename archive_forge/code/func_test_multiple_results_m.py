from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_multiple_results_m():
    expr1 = (x + y) * z
    expr2 = (x - y) * z
    name_expr = ('test', [expr1, expr2])
    result, = codegen(name_expr, 'Octave', header=False, empty=False)
    source = result[1]
    expected = 'function [out1, out2] = test(x, y, z)\n  out1 = z.*(x + y);\n  out2 = z.*(x - y);\nend\n'
    assert source == expected