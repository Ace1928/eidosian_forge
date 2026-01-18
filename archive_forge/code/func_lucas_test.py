from Cryptodome import Random
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.py3compat import iter_range
from Cryptodome.Util.number import sieve_base as _sieve_base_large
def lucas_test(candidate):
    """Perform a Lucas primality test on an integer.

    The test is specified in Section C.3.3 of `FIPS PUB 186-4`__.

    :Parameters:
      candidate : integer
        The number to test for primality.

    :Returns:
      ``Primality.COMPOSITE`` or ``Primality.PROBABLY_PRIME``.

    .. __: http://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf
    """
    if not isinstance(candidate, Integer):
        candidate = Integer(candidate)
    if candidate in (1, 2, 3, 5):
        return PROBABLY_PRIME
    if candidate.is_even() or candidate.is_perfect_square():
        return COMPOSITE

    def alternate():
        value = 5
        while True:
            yield value
            if value > 0:
                value += 2
            else:
                value -= 2
            value = -value
    for D in alternate():
        if candidate in (D, -D):
            continue
        js = Integer.jacobi_symbol(D, candidate)
        if js == 0:
            return COMPOSITE
        if js == -1:
            break
    K = candidate + 1
    r = K.size_in_bits() - 1
    U_i = Integer(1)
    V_i = Integer(1)
    U_temp = Integer(0)
    V_temp = Integer(0)
    for i in iter_range(r - 1, -1, -1):
        U_temp.set(U_i)
        U_temp *= V_i
        U_temp %= candidate
        V_temp.set(U_i)
        V_temp *= U_i
        V_temp *= D
        V_temp.multiply_accumulate(V_i, V_i)
        if V_temp.is_odd():
            V_temp += candidate
        V_temp >>= 1
        V_temp %= candidate
        if K.get_bit(i):
            U_i.set(U_temp)
            U_i += V_temp
            if U_i.is_odd():
                U_i += candidate
            U_i >>= 1
            U_i %= candidate
            V_i.set(V_temp)
            V_i.multiply_accumulate(U_temp, D)
            if V_i.is_odd():
                V_i += candidate
            V_i >>= 1
            V_i %= candidate
        else:
            U_i.set(U_temp)
            V_i.set(V_temp)
    if U_i == 0:
        return PROBABLY_PRIME
    return COMPOSITE