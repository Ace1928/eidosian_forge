from sympy.combinatorics.group_numbers import (is_nilpotent_number,
from sympy.testing.pytest import raises
from sympy import randprime
def test_is_nilpotent_number():
    assert is_nilpotent_number(21) == False
    assert is_nilpotent_number(randprime(1, 30) ** 12) == True
    raises(ValueError, lambda: is_nilpotent_number(-5))