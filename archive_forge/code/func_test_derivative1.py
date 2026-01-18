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
def test_derivative1():
    x, y = map(Symbol, 'xy')
    p, q = map(Wild, 'pq')
    f = Function('f', nargs=1)
    fd = Derivative(f(x), x)
    assert fd.match(p) == {p: fd}
    assert (fd + 1).match(p + 1) == {p: fd}
    assert fd.match(fd) == {}
    assert (3 * fd).match(p * fd) is not None
    assert (3 * fd - 1).match(p * fd + q) == {p: 3, q: -1}