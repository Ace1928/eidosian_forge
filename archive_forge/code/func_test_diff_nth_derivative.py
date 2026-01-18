from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, diff, Subs)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.tensor.array.ndim_array import NDimArray
from sympy.testing.pytest import raises
from sympy.abc import a, b, c, x, y, z
def test_diff_nth_derivative():
    f = Function('f')
    n = Symbol('n', integer=True)
    expr = diff(sin(x), (x, n))
    expr2 = diff(f(x), (x, 2))
    expr3 = diff(f(x), (x, n))
    assert expr.subs(sin(x), cos(-x)) == Derivative(cos(-x), (x, n))
    assert expr.subs(n, 1).doit() == cos(x)
    assert expr.subs(n, 2).doit() == -sin(x)
    assert expr2.subs(Derivative(f(x), x), y) == Derivative(y, x)
    assert expr3 == Derivative(f(x), (x, n))
    assert diff(x, (x, n)) == Piecewise((x, Eq(n, 0)), (1, Eq(n, 1)), (0, True))
    assert diff(2 * x, (x, n)).dummy_eq(Sum(Piecewise((2 * x * factorial(n) / (factorial(y) * factorial(-y + n)), Eq(y, 0) & Eq(Max(0, -y + n), 0)), (2 * factorial(n) / (factorial(y) * factorial(-y + n)), Eq(y, 0) & Eq(Max(0, -y + n), 1)), (0, True)), (y, 0, n)))
    exprm = x * sin(x)
    mul_diff = diff(exprm, (x, n))
    assert isinstance(mul_diff, Sum)
    for i in range(5):
        assert mul_diff.subs(n, i).doit() == exprm.diff((x, i)).expand()
    exprm2 = 2 * y * x * sin(x) * cos(x) * log(x) * exp(x)
    dex = exprm2.diff((x, n))
    assert isinstance(dex, Sum)
    for i in range(7):
        assert dex.subs(n, i).doit().expand() == exprm2.diff((x, i)).expand()
    assert (cos(x) * sin(y)).diff([[x, y, z]]) == NDimArray([-sin(x) * sin(y), cos(x) * cos(y), 0])