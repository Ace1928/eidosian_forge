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
def test_Mul_is_imaginary_real():
    r = Symbol('r', real=True)
    p = Symbol('p', positive=True)
    i1 = Symbol('i1', imaginary=True)
    i2 = Symbol('i2', imaginary=True)
    x = Symbol('x')
    assert I.is_imaginary is True
    assert I.is_real is False
    assert (-I).is_imaginary is True
    assert (-I).is_real is False
    assert (3 * I).is_imaginary is True
    assert (3 * I).is_real is False
    assert (I * I).is_imaginary is False
    assert (I * I).is_real is True
    e = p + p * I
    j = Symbol('j', integer=True, zero=False)
    assert (e ** j).is_real is None
    assert (e ** (2 * j)).is_real is None
    assert (e ** j).is_imaginary is None
    assert (e ** (2 * j)).is_imaginary is None
    assert (e ** (-1)).is_imaginary is False
    assert (e ** 2).is_imaginary
    assert (e ** 3).is_imaginary is False
    assert (e ** 4).is_imaginary is False
    assert (e ** 5).is_imaginary is False
    assert (e ** (-1)).is_real is False
    assert (e ** 2).is_real is False
    assert (e ** 3).is_real is False
    assert (e ** 4).is_real is True
    assert (e ** 5).is_real is False
    assert (e ** 3).is_complex
    assert (r * i1).is_imaginary is None
    assert (r * i1).is_real is None
    assert (x * i1).is_imaginary is None
    assert (x * i1).is_real is None
    assert (i1 * i2).is_imaginary is False
    assert (i1 * i2).is_real is True
    assert (r * i1 * i2).is_imaginary is False
    assert (r * i1 * i2).is_real is True
    nr = Symbol('nr', real=False, complex=True)
    a = Symbol('a', real=True, nonzero=True)
    b = Symbol('b', real=True)
    assert (i1 * nr).is_real is None
    assert (a * nr).is_real is False
    assert (b * nr).is_real is None
    ni = Symbol('ni', imaginary=False, complex=True)
    a = Symbol('a', real=True, nonzero=True)
    b = Symbol('b', real=True)
    assert (i1 * ni).is_real is False
    assert (a * ni).is_real is None
    assert (b * ni).is_real is None