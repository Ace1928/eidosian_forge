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
def test_Mul_is_integer():
    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True)
    nr = Symbol('nr', rational=False)
    ir = Symbol('ir', irrational=True)
    nz = Symbol('nz', integer=True, zero=False)
    e = Symbol('e', even=True)
    o = Symbol('o', odd=True)
    i2 = Symbol('2', prime=True, even=True)
    assert (k / 3).is_integer is None
    assert (nz / 3).is_integer is None
    assert (nr / 3).is_integer is False
    assert (ir / 3).is_integer is False
    assert (x * k * n).is_integer is None
    assert (e / 2).is_integer is True
    assert (e ** 2 / 2).is_integer is True
    assert (2 / k).is_integer is None
    assert (2 / k ** 2).is_integer is None
    assert ((-1) ** k * n).is_integer is True
    assert (3 * k * e / 2).is_integer is True
    assert (2 * k * e / 3).is_integer is None
    assert (e / o).is_integer is None
    assert (o / e).is_integer is False
    assert (o / i2).is_integer is False
    assert Mul(k, 1 / k, evaluate=False).is_integer is None
    assert Mul(2.0, S.Half, evaluate=False).is_integer is None
    assert (2 * sqrt(k)).is_integer is None
    assert (2 * k ** n).is_integer is None
    s = 2 ** 2 ** 2 ** Pow(2, 1000, evaluate=False)
    m = Mul(s, s, evaluate=False)
    assert m.is_integer
    xq = Symbol('xq', rational=True)
    yq = Symbol('yq', rational=True)
    assert (xq * yq).is_integer is None
    e_20161 = Mul(-1, Mul(1, Pow(2, -1, evaluate=False), evaluate=False), evaluate=False)
    assert e_20161.is_integer is not True