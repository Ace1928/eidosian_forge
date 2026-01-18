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
def test_Pow_is_finite():
    xe = Symbol('xe', extended_real=True)
    xr = Symbol('xr', real=True)
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    i = Symbol('i', integer=True)
    assert (xe ** 2).is_finite is None
    assert (xr ** 2).is_finite is True
    assert (xe ** xe).is_finite is None
    assert (xr ** xe).is_finite is None
    assert (xe ** xr).is_finite is None
    assert (xr ** xr).is_finite is None
    assert (p ** xe).is_finite is None
    assert (p ** xr).is_finite is True
    assert (n ** xe).is_finite is None
    assert (n ** xr).is_finite is True
    assert (sin(xe) ** 2).is_finite is True
    assert (sin(xr) ** 2).is_finite is True
    assert (sin(xe) ** xe).is_finite is None
    assert (sin(xr) ** xr).is_finite is None
    assert (sin(xe) ** exp(xe)).is_finite is None
    assert (sin(xr) ** exp(xr)).is_finite is True
    assert (1 / sin(xe)).is_finite is None
    assert (1 / sin(xr)).is_finite is None
    assert (1 / exp(xe)).is_finite is None
    assert (1 / exp(xr)).is_finite is True
    assert (1 / S.Pi).is_finite is True
    assert (1 / (i - 1)).is_finite is None