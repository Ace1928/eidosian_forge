from sympy.core.function import Function
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.powsimp import powsimp
from sympy.simplify.simplify import simplify
from sympy.abc import x, y, n, k
def test_issue_22606():
    fx = Function('f')(x)
    eq = x + gamma(y)
    ans = gammasimp(eq)
    assert gammasimp(eq.subs(x, fx)).subs(fx, x) == ans
    assert gammasimp(eq.subs(x, cos(x))).subs(cos(x), x) == ans
    assert 1 / gammasimp(1 / eq) == ans
    assert gammasimp(fx.subs(x, eq)).args[0] == ans