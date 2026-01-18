import string
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import (diff, expand_func)
from sympy.core import (EulerGamma, TribonacciConstant)
from sympy.core.numbers import (Float, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.numbers import carmichael
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.integers import floor
from sympy.polys.polytools import cancel
from sympy.series.limits import limit, Limit
from sympy.series.order import O
from sympy.functions import (
from sympy.functions.combinatorial.numbers import _nT
from sympy.core.expr import unchanged
from sympy.core.numbers import GoldenRatio, Integer
from sympy.testing.pytest import raises, nocache_fail, warns_deprecated_sympy
from sympy.abc import x
def test_harmonic():
    n = Symbol('n')
    m = Symbol('m')
    assert harmonic(n, 0) == n
    assert harmonic(n).evalf() == harmonic(n)
    assert harmonic(n, 1) == harmonic(n)
    assert harmonic(1, n) == 1
    assert harmonic(0, 1) == 0
    assert harmonic(1, 1) == 1
    assert harmonic(2, 1) == Rational(3, 2)
    assert harmonic(3, 1) == Rational(11, 6)
    assert harmonic(4, 1) == Rational(25, 12)
    assert harmonic(0, 2) == 0
    assert harmonic(1, 2) == 1
    assert harmonic(2, 2) == Rational(5, 4)
    assert harmonic(3, 2) == Rational(49, 36)
    assert harmonic(4, 2) == Rational(205, 144)
    assert harmonic(0, 3) == 0
    assert harmonic(1, 3) == 1
    assert harmonic(2, 3) == Rational(9, 8)
    assert harmonic(3, 3) == Rational(251, 216)
    assert harmonic(4, 3) == Rational(2035, 1728)
    assert harmonic(oo, -1) is S.NaN
    assert harmonic(oo, 0) is oo
    assert harmonic(oo, S.Half) is oo
    assert harmonic(oo, 1) is oo
    assert harmonic(oo, 2) == pi ** 2 / 6
    assert harmonic(oo, 3) == zeta(3)
    assert harmonic(oo, Dummy(negative=True)) is S.NaN
    ip = Dummy(integer=True, positive=True)
    if (1 / ip <= 1) is True:
        assert None, 'delete this if-block and the next line'
    ip = Dummy(even=True, positive=True)
    assert harmonic(oo, 1 / ip) is oo
    assert harmonic(oo, 1 + ip) is zeta(1 + ip)
    assert harmonic(0, m) == 0
    assert harmonic(-1, -1) == 0
    assert harmonic(-1, 0) == -1
    assert harmonic(-1, 1) is S.ComplexInfinity
    assert harmonic(-1, 2) is S.NaN
    assert harmonic(-3, -2) == -5
    assert harmonic(-3, -3) == 9