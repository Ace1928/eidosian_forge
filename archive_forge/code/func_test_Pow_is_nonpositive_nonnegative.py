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
def test_Pow_is_nonpositive_nonnegative():
    x = Symbol('x', real=True)
    k = Symbol('k', integer=True, nonnegative=True)
    l = Symbol('l', integer=True, positive=True)
    n = Symbol('n', even=True)
    m = Symbol('m', odd=True)
    assert (x ** (4 * k)).is_nonnegative is True
    assert (2 ** x).is_nonnegative is True
    assert ((-2) ** x).is_nonnegative is None
    assert ((-2) ** n).is_nonnegative is True
    assert ((-2) ** m).is_nonnegative is False
    assert (k ** 2).is_nonnegative is True
    assert (k ** (-2)).is_nonnegative is None
    assert (k ** k).is_nonnegative is True
    assert (k ** x).is_nonnegative is None
    assert (l ** x).is_nonnegative is True
    assert (l ** x).is_positive is True
    assert ((-k) ** x).is_nonnegative is None
    assert ((-k) ** m).is_nonnegative is None
    assert (2 ** x).is_nonpositive is False
    assert ((-2) ** x).is_nonpositive is None
    assert ((-2) ** n).is_nonpositive is False
    assert ((-2) ** m).is_nonpositive is True
    assert (k ** 2).is_nonpositive is None
    assert (k ** (-2)).is_nonpositive is None
    assert (k ** x).is_nonpositive is None
    assert ((-k) ** x).is_nonpositive is None
    assert ((-k) ** n).is_nonpositive is None
    assert (x ** 2).is_nonnegative is True
    i = symbols('i', imaginary=True)
    assert (i ** 2).is_nonpositive is True
    assert (i ** 4).is_nonpositive is False
    assert (i ** 3).is_nonpositive is False
    assert (I ** i).is_nonnegative is True
    assert (exp(I) ** i).is_nonnegative is True
    assert ((-l) ** n).is_nonnegative is True
    assert ((-l) ** m).is_nonpositive is True
    assert ((-k) ** n).is_nonnegative is None
    assert ((-k) ** m).is_nonpositive is None