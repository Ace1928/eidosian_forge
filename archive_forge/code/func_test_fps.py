from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, asech)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin)
from sympy.functions.special.bessel import airyai
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate
from sympy.series.formal import fps
from sympy.series.order import O
from sympy.series.formal import (rational_algorithm, FormalPowerSeries,
from sympy.testing.pytest import raises, XFAIL, slow
def test_fps():
    assert fps(1) == 1
    assert fps(2, x) == 2
    assert fps(2, x, dir='+') == 2
    assert fps(2, x, dir='-') == 2
    assert fps(1 / x + 1 / x ** 2) == 1 / x + 1 / x ** 2
    assert fps(log(1 + x), hyper=False, rational=False) == log(1 + x)
    f = fps(x ** 2 + x + 1)
    assert isinstance(f, FormalPowerSeries)
    assert f.function == x ** 2 + x + 1
    assert f[0] == 1
    assert f[2] == x ** 2
    assert f.truncate(4) == x ** 2 + x + 1 + O(x ** 4)
    assert f.polynomial() == x ** 2 + x + 1
    f = fps(log(1 + x))
    assert isinstance(f, FormalPowerSeries)
    assert f.function == log(1 + x)
    assert f.subs(x, y) == f
    assert f[:5] == [0, x, -x ** 2 / 2, x ** 3 / 3, -x ** 4 / 4]
    assert f.as_leading_term(x) == x
    assert f.polynomial(6) == x - x ** 2 / 2 + x ** 3 / 3 - x ** 4 / 4 + x ** 5 / 5
    k = f.ak.variables[0]
    assert f.infinite == Sum(-(-1) ** (-k) * x ** k / k, (k, 1, oo))
    ft, s = (f.truncate(n=None), f[:5])
    for i, t in enumerate(ft):
        if i == 5:
            break
        assert s[i] == t
    f = sin(x).fps(x)
    assert isinstance(f, FormalPowerSeries)
    assert f.truncate() == x - x ** 3 / 6 + x ** 5 / 120 + O(x ** 6)
    raises(NotImplementedError, lambda: fps(y * x))
    raises(ValueError, lambda: fps(x, dir=0))