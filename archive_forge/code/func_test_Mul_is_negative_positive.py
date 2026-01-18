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
def test_Mul_is_negative_positive():
    x = Symbol('x', real=True)
    y = Symbol('y', extended_real=False, complex=True)
    z = Symbol('z', zero=True)
    e = 2 * z
    assert e.is_Mul and e.is_positive is False and (e.is_negative is False)
    neg = Symbol('neg', negative=True)
    pos = Symbol('pos', positive=True)
    nneg = Symbol('nneg', nonnegative=True)
    npos = Symbol('npos', nonpositive=True)
    assert neg.is_negative is True
    assert (-neg).is_negative is False
    assert (2 * neg).is_negative is True
    assert (2 * pos)._eval_is_extended_negative() is False
    assert (2 * pos).is_negative is False
    assert pos.is_negative is False
    assert (-pos).is_negative is True
    assert (2 * pos).is_negative is False
    assert (pos * neg).is_negative is True
    assert (2 * pos * neg).is_negative is True
    assert (-pos * neg).is_negative is False
    assert (pos * neg * y).is_negative is False
    assert nneg.is_negative is False
    assert (-nneg).is_negative is None
    assert (2 * nneg).is_negative is False
    assert npos.is_negative is None
    assert (-npos).is_negative is False
    assert (2 * npos).is_negative is None
    assert (nneg * npos).is_negative is None
    assert (neg * nneg).is_negative is None
    assert (neg * npos).is_negative is False
    assert (pos * nneg).is_negative is False
    assert (pos * npos).is_negative is None
    assert (npos * neg * nneg).is_negative is False
    assert (npos * pos * nneg).is_negative is None
    assert (-npos * neg * nneg).is_negative is None
    assert (-npos * pos * nneg).is_negative is False
    assert (17 * npos * neg * nneg).is_negative is False
    assert (17 * npos * pos * nneg).is_negative is None
    assert (neg * npos * pos * nneg).is_negative is False
    assert (x * neg).is_negative is None
    assert (nneg * npos * pos * x * neg).is_negative is None
    assert neg.is_positive is False
    assert (-neg).is_positive is True
    assert (2 * neg).is_positive is False
    assert pos.is_positive is True
    assert (-pos).is_positive is False
    assert (2 * pos).is_positive is True
    assert (pos * neg).is_positive is False
    assert (2 * pos * neg).is_positive is False
    assert (-pos * neg).is_positive is True
    assert (-pos * neg * y).is_positive is False
    assert nneg.is_positive is None
    assert (-nneg).is_positive is False
    assert (2 * nneg).is_positive is None
    assert npos.is_positive is False
    assert (-npos).is_positive is None
    assert (2 * npos).is_positive is False
    assert (nneg * npos).is_positive is False
    assert (neg * nneg).is_positive is False
    assert (neg * npos).is_positive is None
    assert (pos * nneg).is_positive is None
    assert (pos * npos).is_positive is False
    assert (npos * neg * nneg).is_positive is None
    assert (npos * pos * nneg).is_positive is False
    assert (-npos * neg * nneg).is_positive is False
    assert (-npos * pos * nneg).is_positive is None
    assert (17 * npos * neg * nneg).is_positive is None
    assert (17 * npos * pos * nneg).is_positive is False
    assert (neg * npos * pos * nneg).is_positive is None
    assert (x * neg).is_positive is None
    assert (nneg * npos * pos * x * neg).is_positive is None