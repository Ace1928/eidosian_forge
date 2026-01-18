from sympy import abc
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import signsimp
from sympy.testing.pytest import XFAIL
def test_gh_issue_2711():
    x = Symbol('x')
    f = meijerg(((), ()), ((0,), ()), x)
    a = Wild('a')
    b = Wild('b')
    assert f.find(a) == {(S.Zero,), ((), ()), ((S.Zero,), ()), x, S.Zero, (), meijerg(((), ()), ((S.Zero,), ()), x)}
    assert f.find(a + b) == {meijerg(((), ()), ((S.Zero,), ()), x), x, S.Zero}
    assert f.find(a ** 2) == {meijerg(((), ()), ((S.Zero,), ()), x), x}