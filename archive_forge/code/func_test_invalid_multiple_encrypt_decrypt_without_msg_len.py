import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
def test_invalid_multiple_encrypt_decrypt_without_msg_len(self):
    for method_name in ('encrypt', 'decrypt'):
        for assoc_data_present in (True, False):
            cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
            if assoc_data_present:
                cipher.update(self.data)
            method = getattr(cipher, method_name)
            method(self.data)
            self.assertRaises(TypeError, method, self.data)