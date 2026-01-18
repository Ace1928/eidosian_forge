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
def test_divisor_sigma():
    assert [divisor_sigma(k) for k in range(1, 12)] == [1, 3, 4, 7, 6, 12, 8, 15, 13, 18, 12]
    assert [divisor_sigma(k, 2) for k in range(1, 12)] == [1, 5, 10, 21, 26, 50, 50, 85, 91, 130, 122]
    assert divisor_sigma(23450) == 50592
    assert divisor_sigma(23450, 0) == 24
    assert divisor_sigma(23450, 1) == 50592
    assert divisor_sigma(23450, 2) == 730747500
    assert divisor_sigma(23450, 3) == 14666785333344
    a = Symbol('a', prime=True)
    b = Symbol('b', prime=True)
    j = Symbol('j', integer=True, positive=True)
    k = Symbol('k', integer=True, positive=True)
    assert divisor_sigma(a ** j * b ** k) == (a ** (j + 1) - 1) * (b ** (k + 1) - 1) / ((a - 1) * (b - 1))
    assert divisor_sigma(a ** j * b ** k, 2) == (a ** (2 * j + 2) - 1) * (b ** (2 * k + 2) - 1) / ((a ** 2 - 1) * (b ** 2 - 1))
    assert divisor_sigma(a ** j * b ** k, 0) == (j + 1) * (k + 1)
    m = Symbol('m', integer=True)
    k = Symbol('k', integer=True)
    assert divisor_sigma(m)
    assert divisor_sigma(m, k)
    assert divisor_sigma(m).subs(m, 3 ** 10) == 88573
    assert divisor_sigma(m, k).subs([(m, 3 ** 10), (k, 3)]) == 213810021790597
    assert summation(divisor_sigma(m), (m, 1, 11)) == 99