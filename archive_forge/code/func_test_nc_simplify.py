from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import unchanged
from sympy.core.function import (count_ops, diff, expand, expand_multinomial, Function, Derivative)
from sympy.core.mul import Mul, _keep_coeff
from sympy.core import GoldenRatio
from sympy.core.numbers import (E, Float, I, oo, pi, Rational, zoo)
from sympy.core.relational import (Eq, Lt, Gt, Ge, Le)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, sign)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, csch, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, sinc, tan)
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.geometry.polygon import rad
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import (Matrix, eye)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys.polytools import (factor, Poly)
from sympy.simplify.simplify import (besselsimp, hypersimp, inversecombine, logcombine, nsimplify, nthroot, posify, separatevars, signsimp, simplify)
from sympy.solvers.solvers import solve
from sympy.testing.pytest import XFAIL, slow, _both_exp_pow
from sympy.abc import x, y, z, t, a, b, c, d, e, f, g, h, i, n
def test_nc_simplify():
    from sympy.simplify.simplify import nc_simplify
    from sympy.matrices.expressions import MatPow, Identity
    from sympy.core import Pow
    from functools import reduce
    a, b, c, d = symbols('a b c d', commutative=False)
    x = Symbol('x')
    A = MatrixSymbol('A', x, x)
    B = MatrixSymbol('B', x, x)
    C = MatrixSymbol('C', x, x)
    D = MatrixSymbol('D', x, x)
    subst = {a: A, b: B, c: C, d: D}
    funcs = {Add: lambda x, y: x + y, Mul: lambda x, y: x * y}

    def _to_matrix(expr):
        if expr in subst:
            return subst[expr]
        if isinstance(expr, Pow):
            return MatPow(_to_matrix(expr.args[0]), expr.args[1])
        elif isinstance(expr, (Add, Mul)):
            return reduce(funcs[expr.func], [_to_matrix(a) for a in expr.args])
        else:
            return expr * Identity(x)

    def _check(expr, simplified, deep=True, matrix=True):
        assert nc_simplify(expr, deep=deep) == simplified
        assert expand(expr) == expand(simplified)
        if matrix:
            m_simp = _to_matrix(simplified).doit(inv_expand=False)
            assert nc_simplify(_to_matrix(expr), deep=deep) == m_simp
    _check(a * b * a * b * a * b * c * (a * b) ** 3 * c, ((a * b) ** 3 * c) ** 2)
    _check(a * b * (a * b) ** (-2) * a * b, 1)
    _check(a ** 2 * b * a * b * a * b * (a * b) ** (-1), a * (a * b) ** 2, matrix=False)
    _check(b * a * b ** 2 * a * b ** 2 * a * b ** 2, b * (a * b ** 2) ** 3)
    _check(a * b * a ** 2 * b * a ** 2 * b * a ** 3, (a * b * a) ** 3 * a ** 2)
    _check(a ** 2 * b * a ** 4 * b * a ** 4 * b * a ** 2, (a ** 2 * b * a ** 2) ** 3)
    _check(a ** 3 * b * a ** 4 * b * a ** 4 * b * a, a ** 3 * (b * a ** 4) ** 3 * a ** (-3))
    _check(a * b * a * b + a * b * c * x * a * b * c, (a * b) ** 2 + x * (a * b * c) ** 2)
    _check(a * b * a * b * c * a * b * a * b * c, ((a * b) ** 2 * c) ** 2)
    _check(b ** (-1) * a ** (-1) * (a * b) ** 2, a * b)
    _check(a ** (-1) * b * c ** (-1), (c * b ** (-1) * a) ** (-1))
    expr = a ** 3 * b * a ** 4 * b * a ** 4 * b * a ** 2 * b * a ** 2 * (b * a ** 2) ** 2 * b * a ** 2 * b * a ** 2
    for _ in range(10):
        expr *= a * b
    _check(expr, a ** 3 * (b * a ** 4) ** 2 * (b * a ** 2) ** 6 * (a * b) ** 10)
    _check((a * b * a * b) ** 2, (a * b * a * b) ** 2, deep=False)
    _check(a * b * (c * d) ** 2, a * b * (c * d) ** 2)
    expr = b ** (-1) * (a ** (-1) * b ** (-1) - a ** (-1) * c * b ** (-1)) ** (-1) * a ** (-1)
    assert nc_simplify(expr) == (1 - c) ** (-1)
    assert nc_simplify(2 * x ** 2) == 2 * x ** 2