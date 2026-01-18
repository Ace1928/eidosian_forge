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
def test_is_mersenne_prime():
    assert is_mersenne_prime(10) is False
    assert is_mersenne_prime(127) is True
    assert is_mersenne_prime(511) is False
    assert is_mersenne_prime(131071) is True
    assert is_mersenne_prime(2147483647) is True