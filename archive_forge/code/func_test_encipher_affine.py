from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_encipher_affine():
    assert encipher_affine('ABC', (1, 0)) == 'ABC'
    assert encipher_affine('ABC', (1, 1)) == 'BCD'
    assert encipher_affine('ABC', (-1, 0)) == 'AZY'
    assert encipher_affine('ABC', (-1, 1), symbols='ABCD') == 'BAD'
    assert encipher_affine('123', (-1, 1), symbols='1234') == '214'
    assert encipher_affine('ABC', (3, 16)) == 'QTW'
    assert decipher_affine('QTW', (3, 16)) == 'ABC'