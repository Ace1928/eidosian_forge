import random
from bisect import bisect
from itertools import count
from array import array as _array
from sympy.core.function import Function
from sympy.core.singleton import S
from .primetest import isprime
from sympy.utilities.misc import as_int
def prevprime(n):
    """ Return the largest prime smaller than n.

        Notes
        =====

        Potential primes are located at 6*j +/- 1. This
        property is used during searching.

        >>> from sympy import prevprime
        >>> [(i, prevprime(i)) for i in range(10, 15)]
        [(10, 7), (11, 7), (12, 11), (13, 11), (14, 13)]

        See Also
        ========

        nextprime : Return the ith prime greater than n
        primerange : Generates all primes in a given range
    """
    n = _as_int_ceiling(n)
    if n < 3:
        raise ValueError('no preceding primes')
    if n < 8:
        return {3: 2, 4: 3, 5: 3, 6: 5, 7: 5}[n]
    if n <= sieve._list[-1]:
        l, u = sieve.search(n)
        if l == u:
            return sieve[l - 1]
        else:
            return sieve[l]
    nn = 6 * (n // 6)
    if n - nn <= 1:
        n = nn - 1
        if isprime(n):
            return n
        n -= 4
    else:
        n = nn + 1
    while 1:
        if isprime(n):
            return n
        n -= 2
        if isprime(n):
            return n
        n -= 4