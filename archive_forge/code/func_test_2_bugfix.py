import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_2_bugfix(self):
    nonce = unhexlify('EEDDCCBBAA9988776655443322110D')
    key = unhexlify('0F0E0D0C0B0A09080706050403020100')
    A = unhexlify('000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F2021222324252627')
    P = unhexlify('000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F2021222324252627')
    C = unhexlify('07E903BFC49552411ABC865F5ECE60F6FAD1F5A9F14D3070FA2F1308A563207FFE14C1EEA44B22059C7484319D8A2C53C236A7B3')
    mac_len = len(C) - len(P)
    buggy_result = unhexlify('BA015C4E5AE54D76C890AE81BD40DC5703EDC30E8AC2A58BC5D8FA4D61C5BAE6C39BEAC435B2FD56A2A5085C1B135D770C8264B7')
    cipher = AES.new(key, AES.MODE_OCB, nonce=nonce[:-1], mac_len=mac_len)
    cipher.update(A)
    C_out2, tag_out2 = cipher.encrypt_and_digest(P)
    self.assertEqual(buggy_result, C_out2 + tag_out2)