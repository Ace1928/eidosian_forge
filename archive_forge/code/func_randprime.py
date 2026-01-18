import random
from bisect import bisect
from itertools import count
from array import array as _array
from sympy.core.function import Function
from sympy.core.singleton import S
from .primetest import isprime
from sympy.utilities.misc import as_int
def randprime(a, b):
    """ Return a random prime number in the range [a, b).

        Bertrand's postulate assures that
        randprime(a, 2*a) will always succeed for a > 1.

        Examples
        ========

        >>> from sympy import randprime, isprime
        >>> randprime(1, 30) #doctest: +SKIP
        13
        >>> isprime(randprime(1, 30))
        True

        See Also
        ========

        primerange : Generate all primes in a given range

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Bertrand's_postulate

    """
    if a >= b:
        return
    a, b = map(int, (a, b))
    n = random.randint(a - 1, b)
    p = nextprime(n)
    if p >= b:
        p = prevprime(b)
    if p < a:
        raise ValueError('no primes exist in the specified range')
    return p