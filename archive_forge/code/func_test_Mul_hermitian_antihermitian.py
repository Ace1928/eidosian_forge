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
def test_Mul_hermitian_antihermitian():
    xz, yz = symbols('xz, yz', zero=True, antihermitian=True)
    xf, yf = symbols('xf, yf', hermitian=False, antihermitian=False, finite=True)
    xh, yh = symbols('xh, yh', hermitian=True, antihermitian=False, nonzero=True)
    xa, ya = symbols('xa, ya', hermitian=False, antihermitian=True, zero=False, finite=True)
    assert (xz * xh).is_hermitian is True
    assert (xz * xh).is_antihermitian is True
    assert (xz * xa).is_hermitian is True
    assert (xz * xa).is_antihermitian is True
    assert (xf * yf).is_hermitian is None
    assert (xf * yf).is_antihermitian is None
    assert (xh * yh).is_hermitian is True
    assert (xh * yh).is_antihermitian is False
    assert (xh * ya).is_hermitian is False
    assert (xh * ya).is_antihermitian is True
    assert (xa * ya).is_hermitian is True
    assert (xa * ya).is_antihermitian is False
    a = Symbol('a', hermitian=True, zero=False)
    b = Symbol('b', hermitian=True)
    c = Symbol('c', hermitian=False)
    d = Symbol('d', antihermitian=True)
    e1 = Mul(a, b, c, evaluate=False)
    e2 = Mul(b, a, c, evaluate=False)
    e3 = Mul(a, b, c, d, evaluate=False)
    e4 = Mul(b, a, c, d, evaluate=False)
    e5 = Mul(a, c, evaluate=False)
    e6 = Mul(a, c, d, evaluate=False)
    assert e1.is_hermitian is None
    assert e2.is_hermitian is None
    assert e1.is_antihermitian is None
    assert e2.is_antihermitian is None
    assert e3.is_antihermitian is None
    assert e4.is_antihermitian is None
    assert e5.is_antihermitian is None
    assert e6.is_antihermitian is None