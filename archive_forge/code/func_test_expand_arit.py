from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational as R, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.order import O
from sympy.simplify.radsimp import expand_numer
from sympy.core.function import expand, expand_multinomial, expand_power_base
from sympy.testing.pytest import raises
from sympy.core.random import verify_numerically
from sympy.abc import x, y, z
def test_expand_arit():
    a = Symbol('a')
    b = Symbol('b', positive=True)
    c = Symbol('c')
    p = R(5)
    e = (a + b) * c
    assert e == c * (a + b)
    assert e.expand() - a * c - b * c == R(0)
    e = (a + b) * (a + b)
    assert e == (a + b) ** 2
    assert e.expand() == 2 * a * b + a ** 2 + b ** 2
    e = (a + b) * (a + b) ** R(2)
    assert e == (a + b) ** 3
    assert e.expand() == 3 * b * a ** 2 + 3 * a * b ** 2 + a ** 3 + b ** 3
    assert e.expand() == 3 * b * a ** 2 + 3 * a * b ** 2 + a ** 3 + b ** 3
    e = (a + b) * (a + c) * (b + c)
    assert e == (a + c) * (a + b) * (b + c)
    assert e.expand() == 2 * a * b * c + b * a ** 2 + c * a ** 2 + b * c ** 2 + a * c ** 2 + c * b ** 2 + a * b ** 2
    e = (a + R(1)) ** p
    assert e == (1 + a) ** 5
    assert e.expand() == 1 + 5 * a + 10 * a ** 2 + 10 * a ** 3 + 5 * a ** 4 + a ** 5
    e = (a + b + c) * (a + c + p)
    assert e == (5 + a + c) * (a + b + c)
    assert e.expand() == 5 * a + 5 * b + 5 * c + 2 * a * c + b * c + a * b + a ** 2 + c ** 2
    x = Symbol('x')
    s = exp(x * x) - 1
    e = s.nseries(x, 0, 6) / x ** 2
    assert e.expand() == 1 + x ** 2 / 2 + O(x ** 4)
    e = (x * (y + z)) ** (x * (y + z)) * (x + y)
    assert e.expand(power_exp=False, power_base=False) == x * (x * y + x * z) ** (x * y + x * z) + y * (x * y + x * z) ** (x * y + x * z)
    assert e.expand(power_exp=False, power_base=False, deep=False) == x * (x * (y + z)) ** (x * (y + z)) + y * (x * (y + z)) ** (x * (y + z))
    e = x * (x + (y + 1) ** 2)
    assert e.expand(deep=False) == x ** 2 + x * (y + 1) ** 2
    e = (x * (y + z)) ** z
    assert e.expand(power_base=True, mul=True, deep=True) in [x ** z * (y + z) ** z, (x * y + x * z) ** z]
    assert ((2 * y) ** z).expand() == 2 ** z * y ** z
    p = Symbol('p', positive=True)
    assert sqrt(-x).expand().is_Pow
    assert sqrt(-x).expand(force=True) == I * sqrt(x)
    assert ((2 * y * p) ** z).expand() == 2 ** z * p ** z * y ** z
    assert ((2 * y * p * x) ** z).expand() == 2 ** z * p ** z * (x * y) ** z
    assert ((2 * y * p * x) ** z).expand(force=True) == 2 ** z * p ** z * x ** z * y ** z
    assert ((2 * y * p * -pi) ** z).expand() == 2 ** z * pi ** z * p ** z * (-y) ** z
    assert ((2 * y * p * -pi * x) ** z).expand() == 2 ** z * pi ** z * p ** z * (-x * y) ** z
    n = Symbol('n', negative=True)
    m = Symbol('m', negative=True)
    assert ((-2 * x * y * n) ** z).expand() == 2 ** z * (-n) ** z * (x * y) ** z
    assert ((-2 * x * y * n * m) ** z).expand() == 2 ** z * (-m) ** z * (-n) ** z * (-x * y) ** z
    assert sqrt(-2 * x * n) == sqrt(2) * sqrt(-n) * sqrt(x)
    assert (cos(x + y) ** 2).expand(trig=True) in [(-sin(x) * sin(y) + cos(x) * cos(y)) ** 2, sin(x) ** 2 * sin(y) ** 2 - 2 * sin(x) * sin(y) * cos(x) * cos(y) + cos(x) ** 2 * cos(y) ** 2]
    x = Symbol('x')
    W = 1
    for i in range(1, 21):
        W = W * (x - i)
    W = W.expand()
    assert W.has(-1672280820 * x ** 15)