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
def test_log1p_opt():
    x = Symbol('x')
    expr1 = log(x + 1)
    opt1 = optimize(expr1, [log1p_opt])
    assert log1p(x) - opt1 == 0
    assert opt1.rewrite(log) == expr1
    expr2 = log(3 * x + 3)
    opt2 = optimize(expr2, [log1p_opt])
    assert log1p(x) + log(3) == opt2
    assert (opt2.rewrite(log) - expr2).simplify() == 0
    expr3 = log(2 * x + 1)
    opt3 = optimize(expr3, [log1p_opt])
    assert log1p(2 * x) - opt3 == 0
    assert opt3.rewrite(log) == expr3
    expr4 = log(x + 3)
    opt4 = optimize(expr4, [log1p_opt])
    assert str(opt4) == 'log(x + 3)'