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
def test_evalc():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    z = Symbol('z')
    assert ((x + I * y) ** 2).expand(complex=True) == x ** 2 + 2 * I * x * y - y ** 2
    assert expand_complex(z ** (2 * I)) == re((re(z) + I * im(z)) ** (2 * I)) + I * im((re(z) + I * im(z)) ** (2 * I))
    assert expand_complex(z ** (2 * I), deep=False) == I * im(z ** (2 * I)) + re(z ** (2 * I))
    assert exp(I * x) != cos(x) + I * sin(x)
    assert exp(I * x).expand(complex=True) == cos(x) + I * sin(x)
    assert exp(I * x + y).expand(complex=True) == exp(y) * cos(x) + I * sin(x) * exp(y)
    assert sin(I * x).expand(complex=True) == I * sinh(x)
    assert sin(x + I * y).expand(complex=True) == sin(x) * cosh(y) + I * sinh(y) * cos(x)
    assert cos(I * x).expand(complex=True) == cosh(x)
    assert cos(x + I * y).expand(complex=True) == cos(x) * cosh(y) - I * sinh(y) * sin(x)
    assert tan(I * x).expand(complex=True) == tanh(x) * I
    assert tan(x + I * y).expand(complex=True) == sin(2 * x) / (cos(2 * x) + cosh(2 * y)) + I * sinh(2 * y) / (cos(2 * x) + cosh(2 * y))
    assert sinh(I * x).expand(complex=True) == I * sin(x)
    assert sinh(x + I * y).expand(complex=True) == sinh(x) * cos(y) + I * sin(y) * cosh(x)
    assert cosh(I * x).expand(complex=True) == cos(x)
    assert cosh(x + I * y).expand(complex=True) == cosh(x) * cos(y) + I * sin(y) * sinh(x)
    assert tanh(I * x).expand(complex=True) == tan(x) * I
    assert tanh(x + I * y).expand(complex=True) == ((sinh(x) * cosh(x) + I * cos(y) * sin(y)) / (sinh(x) ** 2 + cos(y) ** 2)).expand()