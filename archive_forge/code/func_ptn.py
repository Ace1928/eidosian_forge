import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def ptn(n):
    res = bytearray(n)
    pattern = b''.join([bchr(x) for x in range(0, 251)])
    for base in range(0, n - 251, 251):
        res[base:base + 251] = pattern
    remain = n % 251
    if remain:
        base = n // 251 * 251
        res[base:] = pattern[:remain]
    assert len(res) == n
    return res