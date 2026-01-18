from sympy.ntheory import count_digits, digits, is_palindromic
from sympy.testing.pytest import raises
def test_count_digits():
    assert count_digits(55, 2) == {1: 5, 0: 1}
    assert count_digits(55, 10) == {5: 2}
    n = count_digits(123)
    assert n[4] == 0 and type(n[4]) is int