from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_m_matrix_output_autoname():
    expr = Matrix([[x, x + y, 3]])
    name_expr = ('test', expr)
    result, = codegen(name_expr, 'Octave', header=False, empty=False)
    source = result[1]
    expected = 'function out1 = test(x, y)\n  out1 = [x x + y 3];\nend\n'
    assert source == expected