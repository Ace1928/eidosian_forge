from sympy.combinatorics.group_numbers import (is_nilpotent_number,
from sympy.testing.pytest import raises
from sympy import randprime
def test_is_cyclic_number():
    assert is_cyclic_number(15) == True
    assert is_cyclic_number(randprime(1, 2000) ** 2) == False
    assert is_cyclic_number(randprime(1000, 100000)) == True
    assert is_cyclic_number(4) == False
    raises(ValueError, lambda: is_cyclic_number(-5))