import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
from Cryptodome.Cipher import DES, DES3, ARC2, CAST, Blowfish
@classmethod
def create_test(cls, name, factory, key_size):

    def test_template(self, factory=factory, key_size=key_size):
        cipher = factory.new(get_tag_random('cipher', key_size), factory.MODE_EAX, nonce=b'nonce')
        ct, mac = cipher.encrypt_and_digest(b'plaintext')
        cipher = factory.new(get_tag_random('cipher', key_size), factory.MODE_EAX, nonce=b'nonce')
        pt2 = cipher.decrypt_and_verify(ct, mac)
        self.assertEqual(b'plaintext', pt2)
    setattr(cls, 'test_' + name, test_template)