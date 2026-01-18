import random
from bisect import bisect
from itertools import count
from array import array as _array
from sympy.core.function import Function
from sympy.core.singleton import S
from .primetest import isprime
from sympy.utilities.misc import as_int
def primerange(self, a, b=None):
    """Generate all prime numbers in the range [2, a) or [a, b).

        Examples
        ========

        >>> from sympy import sieve, prime

        All primes less than 19:

        >>> print([i for i in sieve.primerange(19)])
        [2, 3, 5, 7, 11, 13, 17]

        All primes greater than or equal to 7 and less than 19:

        >>> print([i for i in sieve.primerange(7, 19)])
        [7, 11, 13, 17]

        All primes through the 10th prime

        >>> list(sieve.primerange(prime(10) + 1))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        """
    if b is None:
        b = _as_int_ceiling(a)
        a = 2
    else:
        a = max(2, _as_int_ceiling(a))
        b = _as_int_ceiling(b)
    if a >= b:
        return
    self.extend(b)
    i = self.search(a)[1]
    maxi = len(self._list) + 1
    while i < maxi:
        p = self._list[i - 1]
        if p < b:
            yield p
            i += 1
        else:
            return