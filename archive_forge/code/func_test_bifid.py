from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_bifid():
    raises(ValueError, lambda: encipher_bifid('abc', 'b', 'abcde'))
    assert encipher_bifid('abc', 'b', 'abcd') == 'bdb'
    raises(ValueError, lambda: decipher_bifid('bdb', 'b', 'abcde'))
    assert encipher_bifid('bdb', 'b', 'abcd') == 'abc'
    raises(ValueError, lambda: bifid_square('abcde'))
    assert bifid5_square('B') == bifid5_square('BACDEFGHIKLMNOPQRSTUVWXYZ')
    assert bifid6_square('B0') == bifid6_square('B0ACDEFGHIJKLMNOPQRSTUVWXYZ123456789')