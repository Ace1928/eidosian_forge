from io import StringIO
from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_multiple_results_rust():
    expr1 = (x + y) * z
    expr2 = (x - y) * z
    name_expr = ('test', [expr1, expr2])
    result, = codegen(name_expr, 'Rust', header=False, empty=False)
    source = result[1]
    expected = 'fn test(x: f64, y: f64, z: f64) -> (f64, f64) {\n    let out1 = z*(x + y);\n    let out2 = z*(x - y);\n    (out1, out2)\n}\n'
    assert source == expected