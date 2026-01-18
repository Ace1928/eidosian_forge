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
def test_Pow_is_real():
    x = Symbol('x', real=True)
    y = Symbol('y', positive=True)
    assert (x ** 2).is_real is True
    assert (x ** 3).is_real is True
    assert (x ** x).is_real is None
    assert (y ** x).is_real is True
    assert (x ** Rational(1, 3)).is_real is None
    assert (y ** Rational(1, 3)).is_real is True
    assert sqrt(-1 - sqrt(2)).is_real is False
    i = Symbol('i', imaginary=True)
    assert (i ** i).is_real is None
    assert (I ** i).is_extended_real is True
    assert ((-I) ** i).is_extended_real is True
    assert (2 ** i).is_real is None
    assert (2 ** I).is_real is False
    assert (2 ** (-I)).is_real is False
    assert (i ** 2).is_extended_real is True
    assert (i ** 3).is_extended_real is False
    assert (i ** x).is_real is None
    e = Symbol('e', even=True)
    o = Symbol('o', odd=True)
    k = Symbol('k', integer=True)
    assert (i ** e).is_extended_real is True
    assert (i ** o).is_extended_real is False
    assert (i ** k).is_real is None
    assert (i ** (4 * k)).is_extended_real is True
    x = Symbol('x', nonnegative=True)
    y = Symbol('y', nonnegative=True)
    assert im(x ** y).expand(complex=True) is S.Zero
    assert (x ** y).is_real is True
    i = Symbol('i', imaginary=True)
    assert (exp(i) ** I).is_extended_real is True
    assert log(exp(i)).is_imaginary is None
    c = Symbol('c', complex=True)
    assert log(c).is_real is None
    assert log(exp(c)).is_real is None
    n = Symbol('n', negative=False)
    assert log(n).is_real is None
    n = Symbol('n', nonnegative=True)
    assert log(n).is_real is None
    assert sqrt(-I).is_real is False
    i = Symbol('i', integer=True)
    assert (1 / (i - 1)).is_real is None
    assert (1 / (i - 1)).is_extended_real is None
    from sympy.core.parameters import evaluate
    x = S(-1)
    with evaluate(False):
        assert x.is_negative is True
    f = Pow(x, -1)
    with evaluate(False):
        assert f.is_imaginary is False