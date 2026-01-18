from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_gm_public_key():
    assert 323 == gm_public_key(17, 19)[1]
    assert 15 == gm_public_key(3, 5)[1]
    raises(ValueError, lambda: gm_public_key(15, 19))