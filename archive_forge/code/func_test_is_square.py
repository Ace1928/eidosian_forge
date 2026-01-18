from sympy.ntheory.generate import Sieve, sieve
from sympy.ntheory.primetest import (mr, is_lucas_prp, is_square,
from sympy.testing.pytest import slow
from sympy.core.numbers import I
def test_is_square():
    assert [i for i in range(25) if is_square(i)] == [0, 1, 4, 9, 16]
    assert not is_square(60 ** 3)
    assert not is_square(60 ** 5)
    assert not is_square(84 ** 7)
    assert not is_square(105 ** 9)
    assert not is_square(120 ** 3)