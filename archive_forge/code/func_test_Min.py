import itertools as it
from sympy.core.expr import unchanged
from sympy.core.function import Function
from sympy.core.numbers import I, oo, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.external import import_module
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.miscellaneous import (sqrt, cbrt, root, Min,
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.delta_functions import Heaviside
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises, skip, ignore_warnings
def test_Min():
    from sympy.abc import x, y, z
    n = Symbol('n', negative=True)
    n_ = Symbol('n_', negative=True)
    nn = Symbol('nn', nonnegative=True)
    nn_ = Symbol('nn_', nonnegative=True)
    p = Symbol('p', positive=True)
    p_ = Symbol('p_', positive=True)
    np = Symbol('np', nonpositive=True)
    np_ = Symbol('np_', nonpositive=True)
    r = Symbol('r', real=True)
    assert Min(5, 4) == 4
    assert Min(-oo, -oo) is -oo
    assert Min(-oo, n) is -oo
    assert Min(n, -oo) is -oo
    assert Min(-oo, np) is -oo
    assert Min(np, -oo) is -oo
    assert Min(-oo, 0) is -oo
    assert Min(0, -oo) is -oo
    assert Min(-oo, nn) is -oo
    assert Min(nn, -oo) is -oo
    assert Min(-oo, p) is -oo
    assert Min(p, -oo) is -oo
    assert Min(-oo, oo) is -oo
    assert Min(oo, -oo) is -oo
    assert Min(n, n) == n
    assert unchanged(Min, n, np)
    assert Min(np, n) == Min(n, np)
    assert Min(n, 0) == n
    assert Min(0, n) == n
    assert Min(n, nn) == n
    assert Min(nn, n) == n
    assert Min(n, p) == n
    assert Min(p, n) == n
    assert Min(n, oo) == n
    assert Min(oo, n) == n
    assert Min(np, np) == np
    assert Min(np, 0) == np
    assert Min(0, np) == np
    assert Min(np, nn) == np
    assert Min(nn, np) == np
    assert Min(np, p) == np
    assert Min(p, np) == np
    assert Min(np, oo) == np
    assert Min(oo, np) == np
    assert Min(0, 0) == 0
    assert Min(0, nn) == 0
    assert Min(nn, 0) == 0
    assert Min(0, p) == 0
    assert Min(p, 0) == 0
    assert Min(0, oo) == 0
    assert Min(oo, 0) == 0
    assert Min(nn, nn) == nn
    assert unchanged(Min, nn, p)
    assert Min(p, nn) == Min(nn, p)
    assert Min(nn, oo) == nn
    assert Min(oo, nn) == nn
    assert Min(p, p) == p
    assert Min(p, oo) == p
    assert Min(oo, p) == p
    assert Min(oo, oo) is oo
    assert Min(n, n_).func is Min
    assert Min(nn, nn_).func is Min
    assert Min(np, np_).func is Min
    assert Min(p, p_).func is Min
    assert Min() is S.Infinity
    assert Min(x) == x
    assert Min(x, y) == Min(y, x)
    assert Min(x, y, z) == Min(z, y, x)
    assert Min(x, Min(y, z)) == Min(z, y, x)
    assert Min(x, Max(y, -oo)) == Min(x, y)
    assert Min(p, oo, n, p, p, p_) == n
    assert Min(p_, n_, p) == n_
    assert Min(n, oo, -7, p, p, 2) == Min(n, -7)
    assert Min(2, x, p, n, oo, n_, p, 2, -2, -2) == Min(-2, x, n, n_)
    assert Min(0, x, 1, y) == Min(0, x, y)
    assert Min(1000, 100, -100, x, p, n) == Min(n, x, -100)
    assert unchanged(Min, sin(x), cos(x))
    assert Min(sin(x), cos(x)) == Min(cos(x), sin(x))
    assert Min(cos(x), sin(x)).subs(x, 1) == cos(1)
    assert Min(cos(x), sin(x)).subs(x, S.Half) == sin(S.Half)
    raises(ValueError, lambda: Min(cos(x), sin(x)).subs(x, I))
    raises(ValueError, lambda: Min(I))
    raises(ValueError, lambda: Min(I, x))
    raises(ValueError, lambda: Min(S.ComplexInfinity, x))
    assert Min(1, x).diff(x) == Heaviside(1 - x)
    assert Min(x, 1).diff(x) == Heaviside(1 - x)
    assert Min(0, -x, 1 - 2 * x).diff(x) == -Heaviside(x + Min(0, -2 * x + 1)) - 2 * Heaviside(2 * x + Min(0, -x) - 1)
    f = Function('f')
    assert Min(1, 2 * Min(f(1), 2))
    e = Min(0, x)
    assert e.n().args == (0, x)
    m = Min(n, p_, n_, r)
    assert m.is_positive is False
    assert m.is_nonnegative is False
    assert m.is_negative is True
    m = Min(p, p_)
    assert m.is_positive is True
    assert m.is_nonnegative is True
    assert m.is_negative is False
    m = Min(p, nn_, p_)
    assert m.is_positive is None
    assert m.is_nonnegative is True
    assert m.is_negative is False
    m = Min(nn, p, r)
    assert m.is_positive is None
    assert m.is_nonnegative is None
    assert m.is_negative is None