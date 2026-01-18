from sympy.core.function import expand_complex
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re, sign)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
def test_expand_inverse():
    assert (1 / (1 + I)).expand(complex=True) == (1 - I) / 2
    assert ((1 + 2 * I) ** (-2)).expand(complex=True) == (-3 - 4 * I) / 25
    assert ((1 + I) ** (-8)).expand(complex=True) == Rational(1, 16)