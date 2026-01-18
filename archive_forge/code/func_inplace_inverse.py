import sys
from Cryptodome.Util.py3compat import tobytes, is_native_int
from Cryptodome.Util._raw_api import (backend, load_lib,
from ._IntegerBase import IntegerBase
def inplace_inverse(self, modulus):
    """Compute the inverse of this number in the ring of
        modulo integers.

        Raise an exception if no inverse exists.
        """
    if not isinstance(modulus, IntegerGMP):
        modulus = IntegerGMP(modulus)
    comp = _gmp.mpz_cmp(modulus._mpz_p, self._zero_mpz_p)
    if comp == 0:
        raise ZeroDivisionError('Modulus cannot be zero')
    if comp < 0:
        raise ValueError('Modulus must be positive')
    result = _gmp.mpz_invert(self._mpz_p, self._mpz_p, modulus._mpz_p)
    if not result:
        raise ValueError('No inverse value can be computed')
    return self