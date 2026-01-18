from sympy.ntheory import count_digits, digits, is_palindromic
from sympy.testing.pytest import raises
def test_digits():
    assert all([digits(n, 2)[1:] == [int(d) for d in format(n, 'b')] for n in range(20)])
    assert all([digits(n, 8)[1:] == [int(d) for d in format(n, 'o')] for n in range(20)])
    assert all([digits(n, 16)[1:] == [int(d, 16) for d in format(n, 'x')] for n in range(20)])
    assert digits(2345, 34) == [34, 2, 0, 33]
    assert digits(384753, 71) == [71, 1, 5, 23, 4]
    assert digits(93409, 10) == [10, 9, 3, 4, 0, 9]
    assert digits(-92838, 11) == [-11, 6, 3, 8, 2, 9]
    assert digits(35, 10) == [10, 3, 5]
    assert digits(35, 10, 3) == [10, 0, 3, 5]
    assert digits(-35, 10, 4) == [-10, 0, 0, 3, 5]
    raises(ValueError, lambda: digits(2, 2, 1))