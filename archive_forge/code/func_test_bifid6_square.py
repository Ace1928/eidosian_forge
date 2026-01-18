from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_bifid6_square():
    A = bifid6
    f = lambda i, j: symbols(A[6 * i + j])
    M = Matrix(6, 6, f)
    assert bifid6_square('') == M