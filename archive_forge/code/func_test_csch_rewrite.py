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
def test_csch_rewrite():
    x = Symbol('x')
    assert csch(x).rewrite(exp) == 1 / (exp(x) / 2 - exp(-x) / 2) == csch(x).rewrite('tractable')
    assert csch(x).rewrite(cosh) == I / cosh(x + I * pi / 2)
    tanh_half = tanh(S.Half * x)
    assert csch(x).rewrite(tanh) == (1 - tanh_half ** 2) / (2 * tanh_half)
    coth_half = coth(S.Half * x)
    assert csch(x).rewrite(coth) == (coth_half ** 2 - 1) / (2 * coth_half)