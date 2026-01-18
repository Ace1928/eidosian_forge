import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
from Cryptodome.SelfTest.Cipher.test_CBC import BlockChainingTests
def test_aes_128_cfb8(self):
    plaintext = '6bc1bee22e409f96e93d7e117393172aae2d'
    ciphertext = '3b79424c9c0dd436bace9e0ed4586a4f32b9'
    key = '2b7e151628aed2a6abf7158809cf4f3c'
    iv = '000102030405060708090a0b0c0d0e0f'
    key = unhexlify(key)
    iv = unhexlify(iv)
    plaintext = unhexlify(plaintext)
    ciphertext = unhexlify(ciphertext)
    cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=8)
    self.assertEqual(cipher.encrypt(plaintext), ciphertext)
    cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=8)
    self.assertEqual(cipher.decrypt(ciphertext), plaintext)