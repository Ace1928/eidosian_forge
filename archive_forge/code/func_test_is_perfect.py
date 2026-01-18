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
def test_is_perfect():
    assert is_perfect(6) is True
    assert is_perfect(15) is False
    assert is_perfect(28) is True
    assert is_perfect(400) is False
    assert is_perfect(496) is True
    assert is_perfect(8128) is True
    assert is_perfect(10000) is False