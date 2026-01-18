from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.function import (expand_mul, expand_trig)
from sympy.core.numbers import (E, I, Integer, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, acsch, asech, asinh, atanh, cosh, coth, csch, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, cos, cot, sec, sin, tan)
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises
def test_cosh_expansion():
    x, y = symbols('x,y')
    assert cosh(x + y).expand(trig=True) == cosh(x) * cosh(y) + sinh(x) * sinh(y)
    assert cosh(2 * x).expand(trig=True) == cosh(x) ** 2 + sinh(x) ** 2
    assert cosh(3 * x).expand(trig=True).expand() == 3 * sinh(x) ** 2 * cosh(x) + cosh(x) ** 3