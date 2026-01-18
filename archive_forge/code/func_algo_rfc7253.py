import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def algo_rfc7253(keylen, taglen, noncelen):
    """Implement the algorithm at page 18 of RFC 7253"""
    key = bchr(0) * (keylen // 8 - 1) + bchr(taglen)
    C = b''
    for i in range(128):
        S = bchr(0) * i
        N = long_to_bytes(3 * i + 1, noncelen // 8)
        cipher = AES.new(key, AES.MODE_OCB, nonce=N, mac_len=taglen // 8)
        cipher.update(S)
        C += cipher.encrypt(S) + cipher.encrypt() + cipher.digest()
        N = long_to_bytes(3 * i + 2, noncelen // 8)
        cipher = AES.new(key, AES.MODE_OCB, nonce=N, mac_len=taglen // 8)
        C += cipher.encrypt(S) + cipher.encrypt() + cipher.digest()
        N = long_to_bytes(3 * i + 3, noncelen // 8)
        cipher = AES.new(key, AES.MODE_OCB, nonce=N, mac_len=taglen // 8)
        cipher.update(S)
        C += cipher.encrypt() + cipher.digest()
    N = long_to_bytes(385, noncelen // 8)
    cipher = AES.new(key, AES.MODE_OCB, nonce=N, mac_len=taglen // 8)
    cipher.update(C)
    return cipher.encrypt() + cipher.digest()