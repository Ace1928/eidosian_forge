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
def test_Add_Mul_is_integer():
    x = Symbol('x')
    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True)
    nk = Symbol('nk', integer=False)
    nr = Symbol('nr', rational=False)
    nz = Symbol('nz', integer=True, zero=False)
    assert (-nk).is_integer is None
    assert (-nr).is_integer is False
    assert (2 * k).is_integer is True
    assert (-k).is_integer is True
    assert (k + nk).is_integer is False
    assert (k + n).is_integer is True
    assert (k + x).is_integer is None
    assert (k + n * x).is_integer is None
    assert (k + n / 3).is_integer is None
    assert (k + nz / 3).is_integer is None
    assert (k + nr / 3).is_integer is False
    assert ((1 + sqrt(3)) * (-sqrt(3) + 1)).is_integer is not False
    assert (1 + (1 + sqrt(3)) * (-sqrt(3) + 1)).is_integer is not False