import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
def test_aes_128(self):
    key = '2b7e151628aed2a6abf7158809cf4f3c'
    iv = '000102030405060708090a0b0c0d0e0f'
    plaintext = '6bc1bee22e409f96e93d7e117393172a' + 'ae2d8a571e03ac9c9eb76fac45af8e51' + '30c81c46a35ce411e5fbc1191a0a52ef' + 'f69f2445df4f9b17ad2b417be66c3710'
    ciphertext = '7649abac8119b246cee98e9b12e9197d' + '5086cb9b507219ee95db113a917678b2' + '73bed6b8e3c1743b7116e69e22229516' + '3ff1caa1681fac09120eca307586e1a7'
    key = unhexlify(key)
    iv = unhexlify(iv)
    plaintext = unhexlify(plaintext)
    ciphertext = unhexlify(ciphertext)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    self.assertEqual(cipher.encrypt(plaintext), ciphertext)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    self.assertEqual(cipher.decrypt(ciphertext), plaintext)