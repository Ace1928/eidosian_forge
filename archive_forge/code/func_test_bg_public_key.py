from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_bg_public_key():
    assert 5293 == bg_public_key(67, 79)
    assert 713 == bg_public_key(23, 31)
    raises(ValueError, lambda: bg_private_key(13, 17))