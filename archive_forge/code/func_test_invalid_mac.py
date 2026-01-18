import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_invalid_mac(self):
    from Cryptodome.Util.strxor import strxor_c
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    ct, mac = cipher.encrypt_and_digest(self.data)
    invalid_mac = strxor_c(mac, 1)
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    self.assertRaises(ValueError, cipher.decrypt_and_verify, ct, invalid_mac)