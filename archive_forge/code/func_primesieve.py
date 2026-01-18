import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def primesieve(n):
    global sieve_cache, primes_cache, mult_cache
    if n < len(sieve_cache):
        sieve = sieve_cache
        primes = primes_cache[:primes_cache.index(max(sieve)) + 1]
        mult = mult_cache
        return (sieve, primes, mult)
    sieve = [0] * (n + 1)
    mult = [0] * (n + 1)
    primes = list_primes(n)
    for p in primes:
        for k in xrange(p, n + 1, p):
            sieve[k] = p
    for i, p in enumerate(sieve):
        if i >= 2:
            m = 1
            n = i // p
            while not n % p:
                n //= p
                m += 1
            mult[i] = m
    sieve_cache = sieve
    primes_cache = primes
    mult_cache = mult
    return (sieve, primes, mult)