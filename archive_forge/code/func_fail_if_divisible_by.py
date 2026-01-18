import sys
from Cryptodome.Util.py3compat import tobytes, is_native_int
from Cryptodome.Util._raw_api import (backend, load_lib,
from ._IntegerBase import IntegerBase
def fail_if_divisible_by(self, small_prime):
    """Raise an exception if the small prime is a divisor."""
    if is_native_int(small_prime):
        if 0 < small_prime < 65536:
            if _gmp.mpz_divisible_ui_p(self._mpz_p, c_ulong(small_prime)):
                raise ValueError('The value is composite')
            return
        small_prime = IntegerGMP(small_prime)
    if _gmp.mpz_divisible_p(self._mpz_p, small_prime._mpz_p):
        raise ValueError('The value is composite')