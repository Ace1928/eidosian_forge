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
@XFAIL
def test_issue_4883():
    a = Wild('a')
    x = Symbol('x')
    e = [i ** 2 for i in (x - 2, 2 - x)]
    p = [i ** 2 for i in (x - a, a - x)]
    for eq in e:
        for pat in p:
            assert eq.match(pat) == {a: 2}