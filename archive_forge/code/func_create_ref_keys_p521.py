import os
import errno
import warnings
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import bord, tostr, FileNotFoundError
from Cryptodome.Util.asn1 import DerSequence, DerBitString
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Hash import SHAKE128
from Cryptodome.PublicKey import ECC
def create_ref_keys_p521():
    key_len = 66
    key_lines = load_file('ecc_p521.txt').splitlines()
    private_key_d = bytes_to_long(compact(key_lines[2:7]))
    public_key_xy = compact(key_lines[8:17])
    assert bord(public_key_xy[0]) == 4
    public_key_x = bytes_to_long(public_key_xy[1:key_len + 1])
    public_key_y = bytes_to_long(public_key_xy[key_len + 1:])
    return (ECC.construct(curve='P-521', d=private_key_d), ECC.construct(curve='P-521', point_x=public_key_x, point_y=public_key_y))