from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_mutltiprime_rsa_full_example():
    puk = rsa_public_key(2, 3, 5, 7, 11, 13, 7)
    prk = rsa_private_key(2, 3, 5, 7, 11, 13, 7)
    assert puk == (30030, 7)
    assert prk == (30030, 823)
    msg = 10
    encrypted = encipher_rsa(2 * msg - 15, puk)
    assert encrypted == 18065
    decrypted = (decipher_rsa(encrypted, prk) + 15) / 2
    assert decrypted == msg
    puk1 = rsa_public_key(53, 41, 43, 47, 41)
    prk1 = rsa_private_key(53, 41, 43, 47, 41)
    puk2 = rsa_public_key(53, 41, 43, 47, 97)
    prk2 = rsa_private_key(53, 41, 43, 47, 97)
    assert puk1 == (4391633, 41)
    assert prk1 == (4391633, 294041)
    assert puk2 == (4391633, 97)
    assert prk2 == (4391633, 455713)
    msg = 12321
    encrypted = encipher_rsa(encipher_rsa(msg, puk1), puk2)
    assert encrypted == 1081588
    decrypted = decipher_rsa(decipher_rsa(encrypted, prk2), prk1)
    assert decrypted == msg