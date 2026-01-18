import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
from Cryptodome.SelfTest.Cipher.test_CBC import BlockChainingTests
def test_aes_192_cfb128(self):
    plaintext = '6bc1bee22e409f96e93d7e117393172a' + 'ae2d8a571e03ac9c9eb76fac45af8e51' + '30c81c46a35ce411e5fbc1191a0a52ef' + 'f69f2445df4f9b17ad2b417be66c3710'
    ciphertext = 'cdc80d6fddf18cab34c25909c99a4174' + '67ce7f7f81173621961a2b70171d3d7a' + '2e1e8a1dd59b88b1c8e60fed1efac4c9' + 'c05f9f9ca9834fa042ae8fba584b09ff'
    key = '8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b'
    iv = '000102030405060708090a0b0c0d0e0f'
    key = unhexlify(key)
    iv = unhexlify(iv)
    plaintext = unhexlify(plaintext)
    ciphertext = unhexlify(ciphertext)
    cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
    self.assertEqual(cipher.encrypt(plaintext), ciphertext)
    cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
    self.assertEqual(cipher.decrypt(ciphertext), plaintext)