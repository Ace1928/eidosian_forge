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
def test_Add_is_negative_positive():
    x = Symbol('x', real=True)
    k = Symbol('k', negative=True)
    n = Symbol('n', positive=True)
    u = Symbol('u', nonnegative=True)
    v = Symbol('v', nonpositive=True)
    assert (k - 2).is_negative is True
    assert (k + 17).is_negative is None
    assert (-k - 5).is_negative is None
    assert (-k + 123).is_negative is False
    assert (k - n).is_negative is True
    assert (k + n).is_negative is None
    assert (-k - n).is_negative is None
    assert (-k + n).is_negative is False
    assert (k - n - 2).is_negative is True
    assert (k + n + 17).is_negative is None
    assert (-k - n - 5).is_negative is None
    assert (-k + n + 123).is_negative is False
    assert (-2 * k + 123 * n + 17).is_negative is False
    assert (k + u).is_negative is None
    assert (k + v).is_negative is True
    assert (n + u).is_negative is False
    assert (n + v).is_negative is None
    assert (u - v).is_negative is False
    assert (u + v).is_negative is None
    assert (-u - v).is_negative is None
    assert (-u + v).is_negative is None
    assert (u - v + n + 2).is_negative is False
    assert (u + v + n + 2).is_negative is None
    assert (-u - v + n + 2).is_negative is None
    assert (-u + v + n + 2).is_negative is None
    assert (k + x).is_negative is None
    assert (k + x - n).is_negative is None
    assert (k - 2).is_positive is False
    assert (k + 17).is_positive is None
    assert (-k - 5).is_positive is None
    assert (-k + 123).is_positive is True
    assert (k - n).is_positive is False
    assert (k + n).is_positive is None
    assert (-k - n).is_positive is None
    assert (-k + n).is_positive is True
    assert (k - n - 2).is_positive is False
    assert (k + n + 17).is_positive is None
    assert (-k - n - 5).is_positive is None
    assert (-k + n + 123).is_positive is True
    assert (-2 * k + 123 * n + 17).is_positive is True
    assert (k + u).is_positive is None
    assert (k + v).is_positive is False
    assert (n + u).is_positive is True
    assert (n + v).is_positive is None
    assert (u - v).is_positive is None
    assert (u + v).is_positive is None
    assert (-u - v).is_positive is None
    assert (-u + v).is_positive is False
    assert (u - v - n - 2).is_positive is None
    assert (u + v - n - 2).is_positive is None
    assert (-u - v - n - 2).is_positive is None
    assert (-u + v - n - 2).is_positive is False
    assert (n + x).is_positive is None
    assert (n + x - k).is_positive is None
    z = -3 - sqrt(5) + (-sqrt(10) / 2 - sqrt(2) / 2) ** 2
    assert z.is_zero
    z = sqrt(1 + sqrt(3)) + sqrt(3 + 3 * sqrt(3)) - sqrt(10 + 6 * sqrt(3))
    assert z.is_zero