from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_encipher_bifid5():
    assert encipher_bifid5('AB', 'AB') == 'AB'
    assert encipher_bifid5('AB', 'CD') == 'CO'
    assert encipher_bifid5('ab', 'c') == 'CH'
    assert encipher_bifid5('a bc', 'b') == 'BAC'