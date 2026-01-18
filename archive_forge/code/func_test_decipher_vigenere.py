from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_decipher_vigenere():
    assert decipher_vigenere('ABC', 'ABC') == 'AAA'
    assert decipher_vigenere('ABC', 'ABC', symbols='ABCD') == 'AAA'
    assert decipher_vigenere('ABC', 'AB', symbols='ABCD') == 'AAC'
    assert decipher_vigenere('AB', 'ABC', symbols='ABCD') == 'AA'
    assert decipher_vigenere('A', 'ABC', symbols='ABCD') == 'A'