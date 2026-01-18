from sympy.concrete.summations import summation
from sympy.core.containers import Dict
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial as fac
from sympy.core.evalf import bitcount
from sympy.core.numbers import Integer, Rational
from sympy.ntheory import (totient,
from sympy.ntheory.factor_ import (smoothness, smoothness_p, proper_divisors,
from sympy.testing.pytest import raises, slow
from sympy.utilities.iterables import capture
def test_drm():
    assert drm(19, 12) == 7
    assert drm(2718, 10) == 2
    assert drm(0, 15) == 0
    assert drm(234161, 10) == 6
    raises(ValueError, lambda: drm(24, -2))
    raises(ValueError, lambda: drm(11.6, 9))