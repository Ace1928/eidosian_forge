from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import (Derivative, Function, Lambda, Subs)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.core.sympify import SympifyError
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan2, cos, cot, sin, tan)
from sympy.matrices.dense import (Matrix, zeros)
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import RootOf
from sympy.simplify.cse_main import cse
from sympy.simplify.simplify import nsimplify
from sympy.core.basic import _aresame
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import a, x, y, z, t
def test_derivative_subs():
    f = Function('f')
    g = Function('g')
    assert Derivative(f(x), x).subs(f(x), y) != 0
    assert Derivative(f(x), x).subs(f(x), y).xreplace({y: f(x)}) == Derivative(f(x), x)
    assert cse(Derivative(f(x), x) + f(x))[1][0].has(Derivative)
    assert cse(Derivative(f(x, y), x) + Derivative(f(x, y), y))[1][0].has(Derivative)
    eq = Derivative(g(x), g(x))
    assert eq.subs(g, f) == Derivative(f(x), f(x))
    assert eq.subs(g(x), f(x)) == Derivative(f(x), f(x))
    assert eq.subs(g, cos) == Subs(Derivative(y, y), y, cos(x))