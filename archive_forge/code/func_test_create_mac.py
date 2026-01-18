import json
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Hash import CMAC
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
def test_create_mac(self, tv):
    self._id = 'Wycheproof MAC creation Test #' + str(tv.id)
    try:
        tag = CMAC.new(tv.key, tv.msg, ciphermod=AES, mac_len=tv.tag_size).digest()
    except ValueError as e:
        if len(tv.key) not in (16, 24, 32) and 'key length' in str(e):
            return
        raise e
    if tv.valid:
        self.assertEqual(tag, tv.tag)
        self.warn(tv)