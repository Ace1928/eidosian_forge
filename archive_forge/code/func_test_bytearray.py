import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_bytearray(self):
    key_ba = bytearray(self.key_128)
    nonce_ba = bytearray(self.nonce_96)
    header_ba = bytearray(self.data)
    data_ba = bytearray(self.data)
    cipher1 = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    cipher1.update(self.data)
    ct = cipher1.encrypt(self.data) + cipher1.encrypt()
    tag = cipher1.digest()
    cipher2 = AES.new(key_ba, AES.MODE_OCB, nonce=nonce_ba)
    key_ba[:3] = b'\xff\xff\xff'
    nonce_ba[:3] = b'\xff\xff\xff'
    cipher2.update(header_ba)
    header_ba[:3] = b'\xff\xff\xff'
    ct_test = cipher2.encrypt(data_ba) + cipher2.encrypt()
    data_ba[:3] = b'\xff\xff\xff'
    tag_test = cipher2.digest()
    self.assertEqual(ct, ct_test)
    self.assertEqual(tag, tag_test)
    self.assertEqual(cipher1.nonce, cipher2.nonce)
    key_ba = bytearray(self.key_128)
    nonce_ba = bytearray(self.nonce_96)
    header_ba = bytearray(self.data)
    del data_ba
    cipher4 = AES.new(key_ba, AES.MODE_OCB, nonce=nonce_ba)
    key_ba[:3] = b'\xff\xff\xff'
    nonce_ba[:3] = b'\xff\xff\xff'
    cipher4.update(header_ba)
    header_ba[:3] = b'\xff\xff\xff'
    pt_test = cipher4.decrypt_and_verify(bytearray(ct_test), bytearray(tag_test))
    self.assertEqual(self.data, pt_test)