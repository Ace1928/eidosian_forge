from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_m_piecewise_():
    pw = Piecewise((0, x < -1), (x ** 2, x <= 1), (-x + 2, x > 1), (1, True), evaluate=False)
    name_expr = ('pwtest', pw)
    result, = codegen(name_expr, 'Octave', header=False, empty=False)
    source = result[1]
    expected = 'function out1 = pwtest(x)\n  out1 = ((x < -1).*(0) + (~(x < -1)).*( ...\n  (x <= 1).*(x.^2) + (~(x <= 1)).*( ...\n  (x > 1).*(2 - x) + (~(x > 1)).*(1))));\nend\n'
    assert source == expected