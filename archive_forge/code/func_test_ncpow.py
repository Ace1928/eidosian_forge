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
def test_ncpow():
    x = Symbol('x', commutative=False)
    y = Symbol('y', commutative=False)
    z = Symbol('z', commutative=False)
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    assert x ** 2 * y ** 2 != y ** 2 * x ** 2
    assert x ** (-2) * y != y * x ** 2
    assert 2 ** x * 2 ** y != 2 ** (x + y)
    assert 2 ** x * 2 ** y * 2 ** z != 2 ** (x + y + z)
    assert 2 ** x * 2 ** (2 * x) == 2 ** (3 * x)
    assert 2 ** x * 2 ** (2 * x) * 2 ** x == 2 ** (4 * x)
    assert exp(x) * exp(y) != exp(y) * exp(x)
    assert exp(x) * exp(y) * exp(z) != exp(y) * exp(x) * exp(z)
    assert exp(x) * exp(y) * exp(z) != exp(x + y + z)
    assert x ** a * x ** b != x ** (a + b)
    assert x ** a * x ** b * x ** c != x ** (a + b + c)
    assert x ** 3 * x ** 4 == x ** 7
    assert x ** 3 * x ** 4 * x ** 2 == x ** 9
    assert x ** a * x ** (4 * a) == x ** (5 * a)
    assert x ** a * x ** (4 * a) * x ** a == x ** (6 * a)