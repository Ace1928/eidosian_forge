from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_jl_code_argument_order():
    expr = x + y
    routine = make_routine('test', expr, argument_sequence=[z, x, y], language='julia')
    code_gen = JuliaCodeGen()
    output = StringIO()
    code_gen.dump_jl([routine], output, 'test', header=False, empty=False)
    source = output.getvalue()
    expected = 'function test(z, x, y)\n    out1 = x + y\n    return out1\nend\n'
    assert source == expected