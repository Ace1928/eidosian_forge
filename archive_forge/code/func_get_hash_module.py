import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import tobytes, bord, bchr
from Cryptodome.Hash import (SHA1, SHA224, SHA256, SHA384, SHA512,
from Cryptodome.Signature import DSS
from Cryptodome.PublicKey import DSA, ECC
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
def get_hash_module(hash_name):
    if hash_name == 'SHA-512':
        hash_module = SHA512
    elif hash_name == 'SHA-512/224':
        hash_module = SHA512.new(truncate='224')
    elif hash_name == 'SHA-512/256':
        hash_module = SHA512.new(truncate='256')
    elif hash_name == 'SHA-384':
        hash_module = SHA384
    elif hash_name == 'SHA-256':
        hash_module = SHA256
    elif hash_name == 'SHA-224':
        hash_module = SHA224
    elif hash_name == 'SHA-1':
        hash_module = SHA1
    elif hash_name == 'SHA3-224':
        hash_module = SHA3_224
    elif hash_name == 'SHA3-256':
        hash_module = SHA3_256
    elif hash_name == 'SHA3-384':
        hash_module = SHA3_384
    elif hash_name == 'SHA3-512':
        hash_module = SHA3_512
    else:
        raise ValueError('Unknown hash algorithm: ' + hash_name)
    return hash_module