from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, diff, Subs)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.tensor.array.ndim_array import NDimArray
from sympy.testing.pytest import raises
from sympy.abc import a, b, c, x, y, z
def test_diff3():
    p = Rational(5)
    e = a * b + sin(b ** p)
    assert e == a * b + sin(b ** 5)
    assert e.diff(a) == b
    assert e.diff(b) == a + 5 * b ** 4 * cos(b ** 5)
    e = tan(c)
    assert e == tan(c)
    assert e.diff(c) in [cos(c) ** (-2), 1 + sin(c) ** 2 / cos(c) ** 2, 1 + tan(c) ** 2]
    e = c * log(c) - c
    assert e == -c + c * log(c)
    assert e.diff(c) == log(c)
    e = log(sin(c))
    assert e == log(sin(c))
    assert e.diff(c) in [sin(c) ** (-1) * cos(c), cot(c)]
    e = Rational(2) ** a / log(Rational(2))
    assert e == 2 ** a * log(Rational(2)) ** (-1)
    assert e.diff(a) == 2 ** a