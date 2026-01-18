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
def test_Mul_is_even_odd():
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    k = Symbol('k', odd=True)
    n = Symbol('n', odd=True)
    m = Symbol('m', even=True)
    assert (2 * x).is_even is True
    assert (2 * x).is_odd is False
    assert (3 * x).is_even is None
    assert (3 * x).is_odd is None
    assert (k / 3).is_integer is None
    assert (k / 3).is_even is None
    assert (k / 3).is_odd is None
    assert (2 * n).is_even is True
    assert (2 * n).is_odd is False
    assert (2 * m).is_even is True
    assert (2 * m).is_odd is False
    assert (-n).is_even is False
    assert (-n).is_odd is True
    assert (k * n).is_even is False
    assert (k * n).is_odd is True
    assert (k * m).is_even is True
    assert (k * m).is_odd is False
    assert (k * n * m).is_even is True
    assert (k * n * m).is_odd is False
    assert (k * m * x).is_even is True
    assert (k * m * x).is_odd is False
    assert (x / 2).is_integer is None
    assert (k / 2).is_integer is False
    assert (m / 2).is_integer is True
    assert (x * y).is_even is None
    assert (x * x).is_even is None
    assert (x * (x + k)).is_even is True
    assert (x * (x + m)).is_even is None
    assert (x * y).is_odd is None
    assert (x * x).is_odd is None
    assert (x * (x + k)).is_odd is False
    assert (x * (x + m)).is_odd is None
    assert (m ** 2 / 2).is_even
    assert (m ** 2 / 3).is_even is False
    assert (2 / m ** 2).is_odd is False
    assert (2 / m).is_odd is None