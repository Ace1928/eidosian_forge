import tempfile
from sympy.core.numbers import pi, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.assumptions import assuming, Q
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.cfunctions import log2, exp2, expm1, log1p
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.codegen.rewriting import (
from sympy.testing.pytest import XFAIL, skip
from sympy.utilities import lambdify
from sympy.utilities._compilation import compile_link_import_strings, has_c
from sympy.utilities._compilation.util import may_xfail
def test_cosm1_opt():
    x = Symbol('x')
    expr1 = cos(x) - 1
    opt1 = optimize(expr1, [cosm1_opt])
    assert cosm1(x) - opt1 == 0
    assert opt1.rewrite(cos) == expr1
    expr2 = 3 * cos(x) - 3
    opt2 = optimize(expr2, [cosm1_opt])
    assert 3 * cosm1(x) == opt2
    assert opt2.rewrite(cos) == expr2
    expr3 = 3 * cos(x) - 5
    opt3 = optimize(expr3, [cosm1_opt])
    assert 3 * cosm1(x) - 2 == opt3
    assert opt3.rewrite(cos) == expr3
    cosm1_opt_non_opportunistic = FuncMinusOneOptim(cos, cosm1, opportunistic=False)
    assert expr3 == optimize(expr3, [cosm1_opt_non_opportunistic])
    assert opt1 == optimize(expr1, [cosm1_opt_non_opportunistic])
    assert opt2 == optimize(expr2, [cosm1_opt_non_opportunistic])
    expr4 = 3 * cos(x) + log(x) - 3
    opt4 = optimize(expr4, [cosm1_opt])
    assert 3 * cosm1(x) + log(x) == opt4
    assert opt4.rewrite(cos) == expr4
    expr5 = 3 * cos(2 * x) - 3
    opt5 = optimize(expr5, [cosm1_opt])
    assert 3 * cosm1(2 * x) == opt5
    assert opt5.rewrite(cos) == expr5
    expr6 = 2 - 2 * cos(x)
    opt6 = optimize(expr6, [cosm1_opt])
    assert -2 * cosm1(x) == opt6
    assert opt6.rewrite(cos) == expr6