from io import StringIO
from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_multifcns_per_file_w_header():
    name_expr = [('foo', [2 * x, 3 * y]), ('bar', [y ** 2, 4 * y])]
    result = codegen(name_expr, 'Rust', header=True, empty=False)
    assert result[0][0] == 'foo.rs'
    source = result[0][1]
    version_str = 'Code generated with SymPy %s' % sympy.__version__
    version_line = version_str.center(76).rstrip()
    expected = "/*\n *%(version_line)s\n *\n *              See http://www.sympy.org/ for more information.\n *\n *                       This file is part of 'project'\n */\nfn foo(x: f64, y: f64) -> (f64, f64) {\n    let out1 = 2*x;\n    let out2 = 3*y;\n    (out1, out2)\n}\nfn bar(y: f64) -> (f64, f64) {\n    let out1 = y.powi(2);\n    let out2 = 4*y;\n    (out1, out2)\n}\n" % {'version_line': version_line}
    assert source == expected