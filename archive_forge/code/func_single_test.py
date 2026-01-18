from __future__ import print_function
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128, SHA256
from Cryptodome.Util.strxor import strxor
def single_test(self, tv=tv):
    self.description = tv.desc
    cipher = AES.new(tv.key, AES.MODE_GCM, nonce=tv.iv, mac_len=len(tv.tag), use_clmul=self.use_clmul)
    cipher.update(tv.aad)
    if 'FAIL' in tv.others:
        self.assertRaises(ValueError, cipher.decrypt_and_verify, tv.ct, tv.tag)
    else:
        pt = cipher.decrypt_and_verify(tv.ct, tv.tag)
        self.assertEqual(pt, tv.pt)