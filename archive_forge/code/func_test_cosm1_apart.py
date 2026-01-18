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
def test_cosm1_apart():
    x = Symbol('x')
    expr1 = 1 / cos(x) - 1
    opt1 = optimize(expr1, [cosm1_opt])
    assert opt1 == -cosm1(x) / cos(x)
    if scipy:
        _check_num_lambdify(expr1, opt1, {x: S(10) ** (-30)}, 5e-61, lambdify_kw={'modules': 'scipy'})
    expr2 = 2 / cos(x) - 2
    opt2 = optimize(expr2, optims_scipy)
    assert opt2 == -2 * cosm1(x) / cos(x)
    if scipy:
        _check_num_lambdify(expr2, opt2, {x: S(10) ** (-30)}, 1e-60, lambdify_kw={'modules': 'scipy'})
    expr3 = pi / cos(3 * x) - pi
    opt3 = optimize(expr3, [cosm1_opt])
    assert opt3 == -pi * cosm1(3 * x) / cos(3 * x)
    if scipy:
        _check_num_lambdify(expr3, opt3, {x: S(10) ** (-30) / 3}, float(5e-61 * pi), lambdify_kw={'modules': 'scipy'})