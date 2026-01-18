import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
from Cryptodome.SelfTest.Cipher.test_CBC import BlockChainingTests
def test_aes_128_cfb128(self):
    plaintext = '6bc1bee22e409f96e93d7e117393172a' + 'ae2d8a571e03ac9c9eb76fac45af8e51' + '30c81c46a35ce411e5fbc1191a0a52ef' + 'f69f2445df4f9b17ad2b417be66c3710'
    ciphertext = '3b3fd92eb72dad20333449f8e83cfb4a' + 'c8a64537a0b3a93fcde3cdad9f1ce58b' + '26751f67a3cbb140b1808cf187a4f4df' + 'c04b05357c5d1c0eeac4c66f9ff7f2e6'
    key = '2b7e151628aed2a6abf7158809cf4f3c'
    iv = '000102030405060708090a0b0c0d0e0f'
    key = unhexlify(key)
    iv = unhexlify(iv)
    plaintext = unhexlify(plaintext)
    ciphertext = unhexlify(ciphertext)
    cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
    self.assertEqual(cipher.encrypt(plaintext), ciphertext)
    cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
    self.assertEqual(cipher.decrypt(ciphertext), plaintext)