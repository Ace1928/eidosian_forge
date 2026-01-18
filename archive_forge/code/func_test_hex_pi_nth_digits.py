from sympy.core.random import randint
from sympy.ntheory.bbp_pi import pi_hex_digits
from sympy.testing.pytest import raises
def test_hex_pi_nth_digits():
    assert pi_hex_digits(0) == '3243f6a8885a30'
    assert pi_hex_digits(1) == '243f6a8885a308'
    assert pi_hex_digits(10000) == '68ac8fcfb8016c'
    assert pi_hex_digits(13) == '08d313198a2e03'
    assert pi_hex_digits(0, 3) == '324'
    assert pi_hex_digits(0, 0) == ''
    raises(ValueError, lambda: pi_hex_digits(-1))
    raises(ValueError, lambda: pi_hex_digits(3.14))
    n = randint(0, len(dig))
    prec = randint(0, len(dig) - n)
    assert pi_hex_digits(n, prec) == dig[n:n + prec]