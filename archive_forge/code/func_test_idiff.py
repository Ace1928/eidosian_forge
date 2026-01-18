from sympy.core.function import (Derivative, Function)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions import exp, cos, sin, tan, cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry import Point, Point2D, Line, Polygon, Segment, convex_hull,\
from sympy.geometry.util import idiff, closest_points, farthest_points, _ordered_points, are_coplanar
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises
def test_idiff():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    t = Symbol('t', real=True)
    f = Function('f')
    g = Function('g')
    circ = x ** 2 + y ** 2 - 4
    ans = -3 * x * (x ** 2 / y ** 2 + 1) / y ** 3
    assert ans == idiff(circ, y, x, 3), idiff(circ, y, x, 3)
    assert ans == idiff(circ, [y], x, 3)
    assert idiff(circ, y, x, 3) == ans
    explicit = 12 * x / sqrt(-x ** 2 + 4) ** 5
    assert ans.subs(y, solve(circ, y)[0]).equals(explicit)
    assert True in [sol.diff(x, 3).equals(explicit) for sol in solve(circ, y)]
    assert idiff(x + t + y, [y, t], x) == -Derivative(t, x) - 1
    assert idiff(f(x) * exp(f(x)) - x * exp(x), f(x), x) == (x + 1) * exp(x) * exp(-f(x)) / (f(x) + 1)
    assert idiff(f(x) - y * exp(x), [f(x), y], x) == (y + Derivative(y, x)) * exp(x)
    assert idiff(f(x) - y * exp(x), [y, f(x)], x) == -y + Derivative(f(x), x) * exp(-x)
    assert idiff(f(x) - g(x), [f(x), g(x)], x) == Derivative(g(x), x)
    fxy = y - (-10 * (-sin(x) + 1 / x) ** 2 + tan(x) ** 2 + 2 * cosh(x / 10))
    assert idiff(fxy, y, x) == -20 * sin(x) * cos(x) + 2 * tan(x) ** 3 + 2 * tan(x) + sinh(x / 10) / 5 + 20 * cos(x) / x - 20 * sin(x) / x ** 2 + 20 / x ** 3