import re
import sys
import unittest
import binascii
import Cryptodome.Hash
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes
from Cryptodome.Util.strxor import strxor_c
def make_hash_tests(module, module_name, test_data, digest_size, oid=None, extra_params={}):
    tests = []
    for i in range(len(test_data)):
        row = test_data[i]
        expected, input = map(tobytes, row[0:2])
        if len(row) < 3:
            description = repr(input)
        else:
            description = row[2]
        name = '%s #%d: %s' % (module_name, i + 1, description)
        tests.append(HashSelfTest(module, name, expected, input, extra_params))
    name = '%s #%d: digest_size' % (module_name, len(test_data) + 1)
    tests.append(HashDigestSizeSelfTest(module, name, digest_size, extra_params))
    if oid is not None:
        tests.append(HashTestOID(module, oid, extra_params))
    tests.append(ByteArrayTest(module, extra_params))
    tests.append(MemoryViewTest(module, extra_params))
    return tests