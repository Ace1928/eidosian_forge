from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_encipher_hill():
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert encipher_hill('ABCD', A) == 'CFIV'
    A = Matrix(2, 2, [1, 0, 0, 1])
    assert encipher_hill('ABCD', A) == 'ABCD'
    assert encipher_hill('ABCD', A, symbols='ABCD') == 'ABCD'
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert encipher_hill('ABCD', A, symbols='ABCD') == 'CBAB'
    assert encipher_hill('AB', A, symbols='ABCD') == 'CB'
    assert encipher_hill('ABA', A) == 'CFGC'
    assert encipher_hill('ABA', A, pad='Z') == 'CFYV'