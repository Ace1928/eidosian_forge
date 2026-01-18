import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
from Cryptodome.SelfTest.Cipher.test_CBC import BlockChainingTests
def test_des3(self):
    plaintext = 'ac1762037074324fb53ba3596f73656d69746556616c6c6579'
    ciphertext = '9979238528357b90e2e0be549cb0b2d5999b9a4a447e5c5c7d'
    key = '7ade65b460f5ea9be35f9e14aa883a2048e3824aa616c0b2'
    iv = 'cd47e2afb8b7e4b0'
    encrypted_iv = '6a7eef0b58050e8b904a'
    plaintext = unhexlify(plaintext)
    ciphertext = unhexlify(ciphertext)
    key = unhexlify(key)
    iv = unhexlify(iv)
    encrypted_iv = unhexlify(encrypted_iv)
    cipher = DES3.new(key, DES3.MODE_OPENPGP, iv)
    ct = cipher.encrypt(plaintext)
    self.assertEqual(ct[:10], encrypted_iv)
    self.assertEqual(ct[10:], ciphertext)
    cipher = DES3.new(key, DES3.MODE_OPENPGP, encrypted_iv)
    pt = cipher.decrypt(ciphertext)
    self.assertEqual(pt, plaintext)