from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_m_code_argument_order():
    expr = x + y
    routine = make_routine('test', expr, argument_sequence=[z, x, y], language='octave')
    code_gen = OctaveCodeGen()
    output = StringIO()
    code_gen.dump_m([routine], output, 'test', header=False, empty=False)
    source = output.getvalue()
    expected = 'function out1 = test(z, x, y)\n  out1 = x + y;\nend\n'
    assert source == expected