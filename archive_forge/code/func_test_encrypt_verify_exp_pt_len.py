from __future__ import print_function
import unittest
from Cryptodome.PublicKey import RSA
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex
from Cryptodome import Random
from Cryptodome.Cipher import PKCS1_v1_5 as PKCS
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
def test_encrypt_verify_exp_pt_len(self):
    cipher = PKCS.new(self.key1024)
    pt = b'5' * 16
    ct = cipher.encrypt(pt)
    sentinel = b'\xaa' * 16
    pt_A = cipher.decrypt(ct, sentinel, 16)
    self.assertEqual(pt, pt_A)
    pt_B = cipher.decrypt(ct, sentinel, 15)
    self.assertEqual(sentinel, pt_B)
    pt_C = cipher.decrypt(ct, sentinel, 17)
    self.assertEqual(sentinel, pt_C)