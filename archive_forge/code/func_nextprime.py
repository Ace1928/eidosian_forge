import random
from bisect import bisect
from itertools import count
from array import array as _array
from sympy.core.function import Function
from sympy.core.singleton import S
from .primetest import isprime
from sympy.utilities.misc import as_int
def nextprime(n, ith=1):
    """ Return the ith prime greater than n.

        i must be an integer.

        Notes
        =====

        Potential primes are located at 6*j +/- 1. This
        property is used during searching.

        >>> from sympy import nextprime
        >>> [(i, nextprime(i)) for i in range(10, 15)]
        [(10, 11), (11, 13), (12, 13), (13, 17), (14, 17)]
        >>> nextprime(2, ith=2) # the 2nd prime after 2
        5

        See Also
        ========

        prevprime : Return the largest prime smaller than n
        primerange : Generate all primes in a given range

    """
    n = int(n)
    i = as_int(ith)
    if i > 1:
        pr = n
        j = 1
        while 1:
            pr = nextprime(pr)
            j += 1
            if j > i:
                break
        return pr
    if n < 2:
        return 2
    if n < 7:
        return {2: 3, 3: 5, 4: 5, 5: 7, 6: 7}[n]
    if n <= sieve._list[-2]:
        l, u = sieve.search(n)
        if l == u:
            return sieve[u + 1]
        else:
            return sieve[u]
    nn = 6 * (n // 6)
    if nn == n:
        n += 1
        if isprime(n):
            return n
        n += 4
    elif n - nn == 5:
        n += 2
        if isprime(n):
            return n
        n += 4
    else:
        n = nn + 5
    while 1:
        if isprime(n):
            return n
        n += 2
        if isprime(n):
            return n
        n += 4