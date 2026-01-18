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
def test_issue_11518():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    r = sqrt(x ** 2 + y ** 2)
    assert conjugate(r) == r
    s = abs(x + I * y)
    assert conjugate(s) == r