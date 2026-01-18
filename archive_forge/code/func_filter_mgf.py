import unittest
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Util.strxor import strxor
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import SHA1, SHA224, SHA256, SHA384, SHA512
from Cryptodome.PublicKey import RSA
from Cryptodome.Signature import pss
from Cryptodome.Signature import PKCS1_PSS
from Cryptodome.Signature.pss import MGF1
def filter_mgf(group):
    mgf = group['mgf']
    if mgf not in ('MGF1',):
        raise ValueError('Unknown MGF ' + mgf)
    mgf1_hash = get_hash_module(group['mgfSha'])

    def mgf(x, y, mh=mgf1_hash):
        return MGF1(x, y, mh)
    return mgf