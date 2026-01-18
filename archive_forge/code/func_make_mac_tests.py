import re
import sys
import unittest
import binascii
import Cryptodome.Hash
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes
from Cryptodome.Util.strxor import strxor_c
def make_mac_tests(module, module_name, test_data):
    tests = []
    for i, row in enumerate(test_data):
        if len(row) == 4:
            key, data, results, description, params = list(row) + [{}]
        else:
            key, data, results, description, params = row
        name = '%s #%d: %s' % (module_name, i + 1, description)
        tests.append(MACSelfTest(module, name, results, data, key, params))
    return tests