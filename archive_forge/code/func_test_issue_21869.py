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
def test_issue_21869():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    expr = And(Eq(x ** 2, 4), Le(x, y))
    assert expr.simplify() == expr
    expr = And(Eq(x ** 2, 4), Eq(x, 2))
    assert expr.simplify() == Eq(x, 2)
    expr = And(Eq(x ** 3, x ** 2), Eq(x, 1))
    assert expr.simplify() == Eq(x, 1)
    expr = And(Eq(sin(x), x ** 2), Eq(x, 0))
    assert expr.simplify() == Eq(x, 0)
    expr = And(Eq(x ** 3, x ** 2), Eq(x, 2))
    assert expr.simplify() == S.false
    expr = And(Eq(y, x ** 2), Eq(x, 1))
    assert expr.simplify() == And(Eq(y, 1), Eq(x, 1))
    expr = And(Eq(y ** 2, 1), Eq(y, x ** 2), Eq(x, 1))
    assert expr.simplify() == And(Eq(y, 1), Eq(x, 1))
    expr = And(Eq(y ** 2, 4), Eq(y, 2 * x ** 2), Eq(x, 1))
    assert expr.simplify() == And(Eq(y, 2), Eq(x, 1))
    expr = And(Eq(y ** 2, 4), Eq(y, x ** 2), Eq(x, 1))
    assert expr.simplify() == S.false