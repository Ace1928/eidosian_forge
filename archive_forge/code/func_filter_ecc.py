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
def filter_ecc(group):
    if group['key']['curve'] in ('secp224k1', 'secp256k1', 'brainpoolP224r1', 'brainpoolP224t1', 'brainpoolP256r1', 'brainpoolP256t1', 'brainpoolP320r1', 'brainpoolP320t1', 'brainpoolP384r1', 'brainpoolP384t1', 'brainpoolP512r1', 'brainpoolP512t1'):
        return None
    return ECC.import_key(group['keyPem'])