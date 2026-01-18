from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_encipher_railfence():
    assert encipher_railfence('hello world', 2) == 'hlowrdel ol'
    assert encipher_railfence('hello world', 3) == 'horel ollwd'
    assert encipher_railfence('hello world', 4) == 'hwe olordll'