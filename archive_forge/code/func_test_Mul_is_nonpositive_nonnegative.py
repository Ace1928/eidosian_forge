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
def test_Mul_is_nonpositive_nonnegative():
    x = Symbol('x', real=True)
    k = Symbol('k', negative=True)
    n = Symbol('n', positive=True)
    u = Symbol('u', nonnegative=True)
    v = Symbol('v', nonpositive=True)
    assert k.is_nonpositive is True
    assert (-k).is_nonpositive is False
    assert (2 * k).is_nonpositive is True
    assert n.is_nonpositive is False
    assert (-n).is_nonpositive is True
    assert (2 * n).is_nonpositive is False
    assert (n * k).is_nonpositive is True
    assert (2 * n * k).is_nonpositive is True
    assert (-n * k).is_nonpositive is False
    assert u.is_nonpositive is None
    assert (-u).is_nonpositive is True
    assert (2 * u).is_nonpositive is None
    assert v.is_nonpositive is True
    assert (-v).is_nonpositive is None
    assert (2 * v).is_nonpositive is True
    assert (u * v).is_nonpositive is True
    assert (k * u).is_nonpositive is True
    assert (k * v).is_nonpositive is None
    assert (n * u).is_nonpositive is None
    assert (n * v).is_nonpositive is True
    assert (v * k * u).is_nonpositive is None
    assert (v * n * u).is_nonpositive is True
    assert (-v * k * u).is_nonpositive is True
    assert (-v * n * u).is_nonpositive is None
    assert (17 * v * k * u).is_nonpositive is None
    assert (17 * v * n * u).is_nonpositive is True
    assert (k * v * n * u).is_nonpositive is None
    assert (x * k).is_nonpositive is None
    assert (u * v * n * x * k).is_nonpositive is None
    assert k.is_nonnegative is False
    assert (-k).is_nonnegative is True
    assert (2 * k).is_nonnegative is False
    assert n.is_nonnegative is True
    assert (-n).is_nonnegative is False
    assert (2 * n).is_nonnegative is True
    assert (n * k).is_nonnegative is False
    assert (2 * n * k).is_nonnegative is False
    assert (-n * k).is_nonnegative is True
    assert u.is_nonnegative is True
    assert (-u).is_nonnegative is None
    assert (2 * u).is_nonnegative is True
    assert v.is_nonnegative is None
    assert (-v).is_nonnegative is True
    assert (2 * v).is_nonnegative is None
    assert (u * v).is_nonnegative is None
    assert (k * u).is_nonnegative is None
    assert (k * v).is_nonnegative is True
    assert (n * u).is_nonnegative is True
    assert (n * v).is_nonnegative is None
    assert (v * k * u).is_nonnegative is True
    assert (v * n * u).is_nonnegative is None
    assert (-v * k * u).is_nonnegative is None
    assert (-v * n * u).is_nonnegative is True
    assert (17 * v * k * u).is_nonnegative is True
    assert (17 * v * n * u).is_nonnegative is None
    assert (k * v * n * u).is_nonnegative is True
    assert (x * k).is_nonnegative is None
    assert (u * v * n * x * k).is_nonnegative is None