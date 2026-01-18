from Cryptodome import Random
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.py3compat import iter_range
from Cryptodome.Util.number import sieve_base as _sieve_base_large
def miller_rabin_test(candidate, iterations, randfunc=None):
    """Perform a Miller-Rabin primality test on an integer.

    The test is specified in Section C.3.1 of `FIPS PUB 186-4`__.

    :Parameters:
      candidate : integer
        The number to test for primality.
      iterations : integer
        The maximum number of iterations to perform before
        declaring a candidate a probable prime.
      randfunc : callable
        An RNG function where bases are taken from.

    :Returns:
      ``Primality.COMPOSITE`` or ``Primality.PROBABLY_PRIME``.

    .. __: http://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf
    """
    if not isinstance(candidate, Integer):
        candidate = Integer(candidate)
    if candidate in (1, 2, 3, 5):
        return PROBABLY_PRIME
    if candidate.is_even():
        return COMPOSITE
    one = Integer(1)
    minus_one = Integer(candidate - 1)
    if randfunc is None:
        randfunc = Random.new().read
    m = Integer(minus_one)
    a = 0
    while m.is_even():
        m >>= 1
        a += 1
    for i in iter_range(iterations):
        base = 1
        while base in (one, minus_one):
            base = Integer.random_range(min_inclusive=2, max_inclusive=candidate - 2, randfunc=randfunc)
            assert 2 <= base <= candidate - 2
        z = pow(base, m, candidate)
        if z in (one, minus_one):
            continue
        for j in iter_range(1, a):
            z = pow(z, 2, candidate)
            if z == minus_one:
                break
            if z == one:
                return COMPOSITE
        else:
            return COMPOSITE
    return PROBABLY_PRIME