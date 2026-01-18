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
def test_create_expand_pow_optimization():
    cc = lambda x: ccode(optimize(x, [create_expand_pow_optimization(4)]))
    x = Symbol('x')
    assert cc(x ** 4) == 'x*x*x*x'
    assert cc(x ** 4 + x ** 2) == 'x*x + x*x*x*x'
    assert cc(x ** 5 + x ** 4) == 'pow(x, 5) + x*x*x*x'
    assert cc(sin(x) ** 4) == 'pow(sin(x), 4)'
    assert cc(x ** (-4)) == '1.0/(x*x*x*x)'
    assert cc(x ** (-5)) == 'pow(x, -5)'
    assert cc(-x ** 4) == '-(x*x*x*x)'
    assert cc(x ** 4 - x ** 2) == '-(x*x) + x*x*x*x'
    i = Symbol('i', integer=True)
    assert cc(x ** i - x ** 2) == 'pow(x, i) - (x*x)'
    y = Symbol('y', real=True)
    assert cc(Abs(exp(y ** 4))) == 'exp(y*y*y*y)'
    cc2 = lambda x: ccode(optimize(x, [create_expand_pow_optimization(4, base_req=lambda b: b.is_Function)]))
    assert cc2(x ** 3 + sin(x) ** 3) == 'pow(x, 3) + sin(x)*sin(x)*sin(x)'