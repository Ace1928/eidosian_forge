from io import StringIO
from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_complicated_rs_codegen():
    from sympy.functions.elementary.trigonometric import cos, sin, tan
    name_expr = ('testlong', [((sin(x) + cos(y) + tan(z)) ** 3).expand(), cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))])
    result = codegen(name_expr, 'Rust', header=False, empty=False)
    assert result[0][0] == 'testlong.rs'
    source = result[0][1]
    expected = 'fn testlong(x: f64, y: f64, z: f64) -> (f64, f64) {\n    let out1 = x.sin().powi(3) + 3*x.sin().powi(2)*y.cos() + 3*x.sin().powi(2)*z.tan() + 3*x.sin()*y.cos().powi(2) + 6*x.sin()*y.cos()*z.tan() + 3*x.sin()*z.tan().powi(2) + y.cos().powi(3) + 3*y.cos().powi(2)*z.tan() + 3*y.cos()*z.tan().powi(2) + z.tan().powi(3);\n    let out2 = (x + y + z).cos().cos().cos().cos().cos().cos().cos().cos();\n    (out1, out2)\n}\n'
    assert source == expected