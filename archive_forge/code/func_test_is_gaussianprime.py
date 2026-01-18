from sympy.ntheory.generate import Sieve, sieve
from sympy.ntheory.primetest import (mr, is_lucas_prp, is_square,
from sympy.testing.pytest import slow
from sympy.core.numbers import I
def test_is_gaussianprime():
    assert is_gaussian_prime(7 * I)
    assert is_gaussian_prime(7)
    assert is_gaussian_prime(2 + 3 * I)
    assert not is_gaussian_prime(2 + 2 * I)