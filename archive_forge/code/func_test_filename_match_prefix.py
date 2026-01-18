from io import StringIO
from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_filename_match_prefix():
    name_expr = [('foo', [2 * x, 3 * y]), ('bar', [y ** 2, 4 * y])]
    result, = codegen(name_expr, 'Rust', prefix='baz', header=False, empty=False)
    assert result[0] == 'baz.rs'