import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_mac_len(self):
    self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_OCB, nonce=self.nonce_96, mac_len=7)
    self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_OCB, nonce=self.nonce_96, mac_len=16 + 1)
    for mac_len in range(8, 16 + 1):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96, mac_len=mac_len)
        _, mac = cipher.encrypt_and_digest(self.data)
        self.assertEqual(len(mac), mac_len)
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    _, mac = cipher.encrypt_and_digest(self.data)
    self.assertEqual(len(mac), 16)