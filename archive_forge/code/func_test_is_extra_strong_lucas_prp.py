from sympy.ntheory.generate import Sieve, sieve
from sympy.ntheory.primetest import (mr, is_lucas_prp, is_square,
from sympy.testing.pytest import slow
from sympy.core.numbers import I
def test_is_extra_strong_lucas_prp():
    assert is_extra_strong_lucas_prp(4) == False
    assert is_extra_strong_lucas_prp(989) == True
    assert is_extra_strong_lucas_prp(10877) == True
    assert is_extra_strong_lucas_prp(9) == False
    assert is_extra_strong_lucas_prp(16) == False
    assert is_extra_strong_lucas_prp(169) == False