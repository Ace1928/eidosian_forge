from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, comp, nan,
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.polys.polytools import Poly
from sympy.sets.sets import FiniteSet
from sympy.core.parameters import distribute, evaluate
from sympy.core.expr import unchanged
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically
from sympy.functions.elementary.trigonometric import asin
from itertools import product
def test_arit0():
    p = Rational(5)
    e = a * b
    assert e == a * b
    e = a * b + b * a
    assert e == 2 * a * b
    e = a * b + b * a + a * b + p * b * a
    assert e == 8 * a * b
    e = a * b + b * a + a * b + p * b * a + a
    assert e == a + 8 * a * b
    e = a + a
    assert e == 2 * a
    e = a + b + a
    assert e == b + 2 * a
    e = a + b * b + a + b * b
    assert e == 2 * a + 2 * b ** 2
    e = a + Rational(2) + b * b + a + b * b + p
    assert e == 7 + 2 * a + 2 * b ** 2
    e = (a + b * b + a + b * b) * p
    assert e == 5 * (2 * a + 2 * b ** 2)
    e = (a * b * c + c * b * a + b * a * c) * p
    assert e == 15 * a * b * c
    e = (a * b * c + c * b * a + b * a * c) * p - Rational(15) * a * b * c
    assert e == Rational(0)
    e = Rational(50) * (a - a)
    assert e == Rational(0)
    e = b * a - b - a * b + b
    assert e == Rational(0)
    e = a * b + c ** p
    assert e == a * b + c ** 5
    e = a / b
    assert e == a * b ** (-1)
    e = a * 2 * 2
    assert e == 4 * a
    e = 2 + a * 2 / 2
    assert e == 2 + a
    e = 2 - a - 2
    assert e == -a
    e = 2 * a * 2
    assert e == 4 * a
    e = 2 / a / 2
    assert e == a ** (-1)
    e = 2 ** a ** 2
    assert e == 2 ** a ** 2
    e = -(1 + a)
    assert e == -1 - a
    e = S.Half * (1 + a)
    assert e == S.Half + a / 2