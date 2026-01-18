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
def test_mersenne_prime_exponent():
    assert mersenne_prime_exponent(1) == 2
    assert mersenne_prime_exponent(4) == 7
    assert mersenne_prime_exponent(10) == 89
    assert mersenne_prime_exponent(25) == 21701
    raises(ValueError, lambda: mersenne_prime_exponent(52))
    raises(ValueError, lambda: mersenne_prime_exponent(0))