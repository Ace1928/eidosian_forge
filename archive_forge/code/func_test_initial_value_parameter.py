import unittest
from binascii import hexlify, unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128, SHA256
from Cryptodome.Util import Counter
def test_initial_value_parameter(self):
    cipher1 = AES.new(self.key_128, AES.MODE_CTR, nonce=self.nonce_64, initial_value=65535)
    counter = Counter.new(64, prefix=self.nonce_64, initial_value=65535)
    cipher2 = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
    pt = get_tag_random('plaintext', 65536)
    self.assertEqual(cipher1.encrypt(pt), cipher2.encrypt(pt))
    cipher1 = AES.new(self.key_128, AES.MODE_CTR, initial_value=65535)
    counter = Counter.new(64, prefix=cipher1.nonce, initial_value=65535)
    cipher2 = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
    pt = get_tag_random('plaintext', 65536)
    self.assertEqual(cipher1.encrypt(pt), cipher2.encrypt(pt))
    self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR, counter=self.ctr_128, initial_value=0)