from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_lfsr_connection_polynomial():
    F = FF(2)
    x = symbols('x')
    s = lfsr_sequence([F(1), F(0)], [F(0), F(1)], 5)
    assert lfsr_connection_polynomial(s) == x ** 2 + 1
    s = lfsr_sequence([F(1), F(1)], [F(0), F(1)], 5)
    assert lfsr_connection_polynomial(s) == x ** 2 + x + 1