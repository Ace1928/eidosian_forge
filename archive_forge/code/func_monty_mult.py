import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (create_string_buffer,
from Cryptodome.Math._IntegerCustom import _raw_montgomery
def monty_mult(term1, term2, modulus):
    if term1 >= modulus:
        term1 %= modulus
    if term2 >= modulus:
        term2 %= modulus
    modulus_b = long_to_bytes(modulus)
    numbers_len = len(modulus_b)
    term1_b = long_to_bytes(term1, numbers_len)
    term2_b = long_to_bytes(term2, numbers_len)
    out = create_string_buffer(numbers_len)
    error = _raw_montgomery.monty_multiply(out, term1_b, term2_b, modulus_b, c_size_t(numbers_len))
    if error == 17:
        raise ExceptionModulus()
    if error:
        raise ValueError('monty_multiply() failed with error: %d' % error)
    return get_raw_buffer(out)