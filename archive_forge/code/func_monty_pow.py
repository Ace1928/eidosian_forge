import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.py3compat import *
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Hash import SHAKE128
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math._IntegerCustom import _raw_montgomery
from Cryptodome.Random.random import StrongRandom
def monty_pow(base, exp, modulus):
    max_len = len(long_to_bytes(max(base, exp, modulus)))
    base_b, exp_b, modulus_b = [long_to_bytes(x, max_len) for x in (base, exp, modulus)]
    out = create_string_buffer(max_len)
    error = _raw_montgomery.monty_pow(out, base_b, exp_b, modulus_b, c_size_t(max_len), c_ulonglong(32))
    if error == 17:
        raise ExceptionModulus()
    if error:
        raise ValueError('monty_pow failed with error: %d' % error)
    result = bytes_to_long(get_raw_buffer(out))
    return result