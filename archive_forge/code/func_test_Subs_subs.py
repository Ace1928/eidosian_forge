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
def test_Subs_subs():
    assert Subs(x * y, x, x).subs(x, y) == Subs(x * y, x, y)
    assert Subs(x * y, x, x + 1).subs(x, y) == Subs(x * y, x, y + 1)
    assert Subs(x * y, y, x + 1).subs(x, y) == Subs(y ** 2, y, y + 1)
    a = Subs(x * y * z, (y, x, z), (x + 1, x + z, x))
    b = Subs(x * y * z, (y, x, z), (x + 1, y + z, y))
    assert a.subs(x, y) == b and a.doit().subs(x, y) == a.subs(x, y).doit()
    f = Function('f')
    g = Function('g')
    assert Subs(2 * f(x, y) + g(x), f(x, y), 1).subs(y, 2) == Subs(2 * f(x, y) + g(x), (f(x, y), y), (1, 2))