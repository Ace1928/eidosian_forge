import math
import sys
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import iter_range
import struct
import warnings
def getStrongPrime(N, e=0, false_positive_prob=1e-06, randfunc=None):
    """
    Return a random strong *N*-bit prime number.
    In this context, *p* is a strong prime if *p-1* and *p+1* have at
    least one large prime factor.

    Args:
        N (integer): the exact length of the strong prime.
          It must be a multiple of 128 and > 512.
        e (integer): if provided, the returned prime (minus 1)
          will be coprime to *e* and thus suitable for RSA where
          *e* is the public exponent.
        false_positive_prob (float):
          The statistical probability for the result not to be actually a
          prime. It defaults to 10\\ :sup:`-6`.
          Note that the real probability of a false-positive is far less. This is
          just the mathematically provable limit.
        randfunc (callable):
          A function that takes a parameter *N* and that returns
          a random byte string of such length.
          If omitted, :func:`Cryptodome.Random.get_random_bytes` is used.
    Return:
        The new strong prime.

    .. deprecated:: 3.0
        This function is for internal use only and may be renamed or removed in
        the future.
    """
    if randfunc is None:
        randfunc = Random.get_random_bytes
    if _fastmath is not None:
        return _fastmath.getStrongPrime(long(N), long(e), false_positive_prob, randfunc)
    if N < 512 or N % 128 != 0:
        raise ValueError('bits must be multiple of 128 and > 512')
    rabin_miller_rounds = int(math.ceil(-math.log(false_positive_prob) / math.log(4)))
    x = N - 512 >> 7
    lower_bound = 14142135623730950489 * 2 ** (511 + 128 * x) // 10000000000000000000
    upper_bound = (1 << 512 + 128 * x) - 1
    X = getRandomRange(lower_bound, upper_bound, randfunc)
    p = [0, 0]
    for i in (0, 1):
        y = getRandomNBitInteger(101, randfunc)
        field = [0] * 5 * len(sieve_base)
        for prime in sieve_base:
            offset = y % prime
            for j in iter_range((prime - offset) % prime, len(field), prime):
                field[j] = 1
        result = 0
        for j in range(len(field)):
            composite = field[j]
            if composite:
                continue
            tmp = y + j
            result = _rabinMillerTest(tmp, rabin_miller_rounds)
            if result > 0:
                p[i] = tmp
                break
        if result == 0:
            raise RuntimeError("Couln't find prime in field. Developer: Increase field_size")
    tmp1 = inverse(p[1], p[0]) * p[1]
    tmp2 = inverse(p[0], p[1]) * p[0]
    R = tmp1 - tmp2
    increment = p[0] * p[1]
    X = X + (R - X % increment)
    while 1:
        is_possible_prime = 1
        for prime in sieve_base:
            if X % prime == 0:
                is_possible_prime = 0
                break
        if e and is_possible_prime:
            if e & 1:
                if GCD(e, X - 1) != 1:
                    is_possible_prime = 0
            elif GCD(e, (X - 1) // 2) != 1:
                is_possible_prime = 0
        if is_possible_prime:
            result = _rabinMillerTest(X, rabin_miller_rounds)
            if result > 0:
                break
        X += increment
        if X >= 1 << N:
            raise RuntimeError("Couln't find prime in field. Developer: Increase field_size")
    return X