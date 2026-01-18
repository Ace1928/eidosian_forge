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
def test_Pow_is_integer():
    x = Symbol('x')
    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True, nonnegative=True)
    m = Symbol('m', integer=True, positive=True)
    assert (k ** 2).is_integer is True
    assert (k ** (-2)).is_integer is None
    assert ((m + 1) ** (-2)).is_integer is False
    assert (m ** (-1)).is_integer is None
    assert (2 ** k).is_integer is None
    assert (2 ** (-k)).is_integer is None
    assert (2 ** n).is_integer is True
    assert (2 ** (-n)).is_integer is None
    assert (2 ** m).is_integer is True
    assert (2 ** (-m)).is_integer is False
    assert (x ** 2).is_integer is None
    assert (2 ** x).is_integer is None
    assert (k ** n).is_integer is True
    assert (k ** (-n)).is_integer is None
    assert (k ** x).is_integer is None
    assert (x ** k).is_integer is None
    assert (k ** (n * m)).is_integer is True
    assert (k ** (-n * m)).is_integer is None
    assert sqrt(3).is_integer is False
    assert sqrt(0.3).is_integer is False
    assert Pow(3, 2, evaluate=False).is_integer is True
    assert Pow(3, 0, evaluate=False).is_integer is True
    assert Pow(3, -2, evaluate=False).is_integer is False
    assert Pow(S.Half, 3, evaluate=False).is_integer is False
    assert Pow(3, S.Half, evaluate=False).is_integer is False
    assert Pow(3, S.Half, evaluate=False).is_integer is False
    assert Pow(4, S.Half, evaluate=False).is_integer is True
    assert Pow(S.Half, -2, evaluate=False).is_integer is True
    assert ((-1) ** k).is_integer
    x = Symbol('x', real=True, integer=False)
    assert (x ** 2).is_integer is None
    x = Symbol('x', positive=True)
    assert (1 / (x + 1)).is_integer is False
    assert (1 / (-x - 1)).is_integer is False
    assert (-1 / (x + 1)).is_integer is False
    assert (x ** 2 / 2).is_integer is None
    k = Symbol('k', even=True)
    assert (k ** 3 / 2).is_integer
    assert (k ** 3 / 8).is_integer
    assert (k ** 3 / 16).is_integer is None
    assert (2 / k).is_integer is None
    assert (2 / k ** 2).is_integer is False
    o = Symbol('o', odd=True)
    assert (k / o).is_integer is None
    o = Symbol('o', odd=True, prime=True)
    assert (k / o).is_integer is False