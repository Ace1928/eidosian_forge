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
def test_mul_zero_detection():
    nz = Dummy(real=True, zero=False)
    r = Dummy(extended_real=True)
    c = Dummy(real=False, complex=True)
    c2 = Dummy(real=False, complex=True)
    i = Dummy(imaginary=True)
    e = nz * r * c
    assert e.is_imaginary is None
    assert e.is_extended_real is None
    e = nz * c
    assert e.is_imaginary is None
    assert e.is_extended_real is False
    e = nz * i * c
    assert e.is_imaginary is False
    assert e.is_extended_real is None
    e = nz * i * c * c2
    assert e.is_imaginary is None
    assert e.is_extended_real is None

    def test(z, b, e):
        if z.is_zero and b.is_finite:
            assert e.is_extended_real and e.is_zero
        else:
            assert e.is_extended_real is None
            if b.is_finite:
                if z.is_zero:
                    assert e.is_zero
                else:
                    assert e.is_zero is None
            elif b.is_finite is False:
                if z.is_zero is None:
                    assert e.is_zero is None
                else:
                    assert e.is_zero is False
    for iz, ib in product(*[[True, False, None]] * 2):
        z = Dummy('z', nonzero=iz)
        b = Dummy('f', finite=ib)
        e = Mul(z, b, evaluate=False)
        test(z, b, e)
        z = Dummy('nz', nonzero=iz)
        b = Dummy('f', finite=ib)
        e = Mul(b, z, evaluate=False)
        test(z, b, e)

    def test(z, b, e):
        if z.is_zero and (not b.is_finite):
            assert e.is_extended_real is None
        else:
            assert e.is_extended_real is True
    for iz, ib in product(*[[True, False, None]] * 2):
        z = Dummy('z', nonzero=iz, extended_real=True)
        b = Dummy('b', finite=ib, extended_real=True)
        e = Mul(z, b, evaluate=False)
        test(z, b, e)
        z = Dummy('z', nonzero=iz, extended_real=True)
        b = Dummy('b', finite=ib, extended_real=True)
        e = Mul(b, z, evaluate=False)
        test(z, b, e)