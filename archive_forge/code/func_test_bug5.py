from sympy.calculus.util import AccumBounds
from sympy.core.function import (Derivative, PoleError)
from sympy.core.numbers import (E, I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, cot, sin, tan)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.abc import x, y, z
from sympy.testing.pytest import raises, XFAIL
def test_bug5():
    w = Symbol('w')
    l = Symbol('l')
    e = (-log(w) + log(1 + w * log(x))) ** (-2) * w ** (-2) * ((-log(w) + log(1 + x * w)) * (-log(w) + log(1 + w * log(x))) * w - x * (-log(w) + log(1 + w * log(x))) * w)
    assert e.nseries(w, n=0, logx=l) == x / w / l + 1 / w + O(1, w)
    assert e.nseries(w, n=1, logx=l) == x / w / l + 1 / w - x / l + 1 / l * log(x) + x * log(x) / l ** 2 + O(w)