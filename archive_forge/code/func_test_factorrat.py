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
def test_factorrat():
    assert str(factorrat(S(12) / 1, visual=True)) == '2**2*3**1'
    assert str(factorrat(Rational(1, 1), visual=True)) == '1'
    assert str(factorrat(S(25) / 14, visual=True)) == '5**2/(2*7)'
    assert str(factorrat(Rational(25, 14), visual=True)) == '5**2/(2*7)'
    assert str(factorrat(S(-25) / 14 / 9, visual=True)) == '-1*5**2/(2*3**2*7)'
    assert factorrat(S(12) / 1, multiple=True) == [2, 2, 3]
    assert factorrat(Rational(1, 1), multiple=True) == []
    assert factorrat(S(25) / 14, multiple=True) == [Rational(1, 7), S.Half, 5, 5]
    assert factorrat(Rational(25, 14), multiple=True) == [Rational(1, 7), S.Half, 5, 5]
    assert factorrat(Rational(12, 1), multiple=True) == [2, 2, 3]
    assert factorrat(S(-25) / 14 / 9, multiple=True) == [-1, Rational(1, 7), Rational(1, 3), Rational(1, 3), S.Half, 5, 5]