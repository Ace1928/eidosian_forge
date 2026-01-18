from sympy.core.numbers import (E, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core import Add, Mul, Pow
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x
def test_AccumBounds_pow():
    assert B(0, 2) ** 2 == B(0, 4)
    assert B(-1, 1) ** 2 == B(0, 1)
    assert B(1, 2) ** 2 == B(1, 4)
    assert B(-1, 2) ** 3 == B(-1, 8)
    assert B(-1, 1) ** 0 == 1
    assert B(1, 2) ** Rational(5, 2) == B(1, 4 * sqrt(2))
    assert B(0, 2) ** S.Half == B(0, sqrt(2))
    neg = Symbol('neg', negative=True)
    assert unchanged(Pow, B(neg, 1), S.Half)
    nn = Symbol('nn', nonnegative=True)
    assert B(nn, nn + 1) ** S.Half == B(sqrt(nn), sqrt(nn + 1))
    assert B(nn, nn + 1) ** nn == B(nn ** nn, (nn + 1) ** nn)
    assert unchanged(Pow, B(nn, nn + 1), x)
    i = Symbol('i', integer=True)
    assert B(1, 2) ** i == B(Min(1, 2 ** i), Max(1, 2 ** i))
    i = Symbol('i', integer=True, nonnegative=True)
    assert B(1, 2) ** i == B(1, 2 ** i)
    assert B(0, 1) ** i == B(0 ** i, 1)
    assert B(1, 5) ** (-2) == B(Rational(1, 25), 1)
    assert B(-1, 3) ** (-2) == B(0, oo)
    assert B(0, 2) ** (-3) == B(Rational(1, 8), oo)
    assert B(-2, 0) ** (-3) == B(-oo, -Rational(1, 8))
    assert B(0, 2) ** (-2) == B(Rational(1, 4), oo)
    assert B(-1, 2) ** (-3) == B(-oo, oo)
    assert B(-3, -2) ** (-3) == B(Rational(-1, 8), Rational(-1, 27))
    assert B(-3, -2) ** (-2) == B(Rational(1, 9), Rational(1, 4))
    assert B(0, oo) ** S.Half == B(0, oo)
    assert B(-oo, 0) ** (-2) == B(0, oo)
    assert B(-2, 0) ** (-2) == B(Rational(1, 4), oo)
    assert B(Rational(1, 3), S.Half) ** oo is S.Zero
    assert B(0, S.Half) ** oo is S.Zero
    assert B(S.Half, 1) ** oo == B(0, oo)
    assert B(0, 1) ** oo == B(0, oo)
    assert B(2, 3) ** oo is oo
    assert B(1, 2) ** oo == B(0, oo)
    assert B(S.Half, 3) ** oo == B(0, oo)
    assert B(Rational(-1, 3), Rational(-1, 4)) ** oo is S.Zero
    assert B(-1, Rational(-1, 2)) ** oo is S.NaN
    assert B(-3, -2) ** oo is zoo
    assert B(-2, -1) ** oo is S.NaN
    assert B(-2, Rational(-1, 2)) ** oo is S.NaN
    assert B(Rational(-1, 2), S.Half) ** oo is S.Zero
    assert B(Rational(-1, 2), 1) ** oo == B(0, oo)
    assert B(Rational(-2, 3), 2) ** oo == B(0, oo)
    assert B(-1, 1) ** oo == B(-oo, oo)
    assert B(-1, S.Half) ** oo == B(-oo, oo)
    assert B(-1, 2) ** oo == B(-oo, oo)
    assert B(-2, S.Half) ** oo == B(-oo, oo)
    assert B(1, 2) ** x == Pow(B(1, 2), x, evaluate=False)
    assert B(2, 3) ** (-oo) is S.Zero
    assert B(0, 2) ** (-oo) == B(0, oo)
    assert B(-1, 2) ** (-oo) == B(-oo, oo)
    assert (tan(x) ** sin(2 * x)).subs(x, B(0, pi / 2)) == Pow(B(-oo, oo), B(0, 1))