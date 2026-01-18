from io import StringIO
from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_empty_rust_code():
    code_gen = RustCodeGen()
    output = StringIO()
    code_gen.dump_rs([], output, 'file', header=False, empty=False)
    source = output.getvalue()
    assert source == ''