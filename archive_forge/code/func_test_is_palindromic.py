from sympy.ntheory import count_digits, digits, is_palindromic
from sympy.testing.pytest import raises
def test_is_palindromic():
    assert is_palindromic(-11)
    assert is_palindromic(11)
    assert is_palindromic(81, 8)
    assert not is_palindromic(123)