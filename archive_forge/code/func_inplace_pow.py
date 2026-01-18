import sys
from Cryptodome.Util.py3compat import tobytes, is_native_int
from Cryptodome.Util._raw_api import (backend, load_lib,
from ._IntegerBase import IntegerBase
def inplace_pow(self, exponent, modulus=None):
    if modulus is None:
        if exponent < 0:
            raise ValueError('Exponent must not be negative')
        if exponent > 256:
            raise ValueError('Exponent is too big')
        _gmp.mpz_pow_ui(self._mpz_p, self._mpz_p, c_ulong(int(exponent)))
    else:
        if not isinstance(modulus, IntegerGMP):
            modulus = IntegerGMP(modulus)
        if not modulus:
            raise ZeroDivisionError('Division by zero')
        if modulus.is_negative():
            raise ValueError('Modulus must be positive')
        if is_native_int(exponent):
            if exponent < 0:
                raise ValueError('Exponent must not be negative')
            if exponent < 65536:
                _gmp.mpz_powm_ui(self._mpz_p, self._mpz_p, c_ulong(exponent), modulus._mpz_p)
                return self
            exponent = IntegerGMP(exponent)
        elif exponent.is_negative():
            raise ValueError('Exponent must not be negative')
        _gmp.mpz_powm(self._mpz_p, self._mpz_p, exponent._mpz_p, modulus._mpz_p)
    return self