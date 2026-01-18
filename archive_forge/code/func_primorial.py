import random
from bisect import bisect
from itertools import count
from array import array as _array
from sympy.core.function import Function
from sympy.core.singleton import S
from .primetest import isprime
from sympy.utilities.misc import as_int
def primorial(n, nth=True):
    """
    Returns the product of the first n primes (default) or
    the primes less than or equal to n (when ``nth=False``).

    Examples
    ========

    >>> from sympy.ntheory.generate import primorial, primerange
    >>> from sympy import factorint, Mul, primefactors, sqrt
    >>> primorial(4) # the first 4 primes are 2, 3, 5, 7
    210
    >>> primorial(4, nth=False) # primes <= 4 are 2 and 3
    6
    >>> primorial(1)
    2
    >>> primorial(1, nth=False)
    1
    >>> primorial(sqrt(101), nth=False)
    210

    One can argue that the primes are infinite since if you take
    a set of primes and multiply them together (e.g. the primorial) and
    then add or subtract 1, the result cannot be divided by any of the
    original factors, hence either 1 or more new primes must divide this
    product of primes.

    In this case, the number itself is a new prime:

    >>> factorint(primorial(4) + 1)
    {211: 1}

    In this case two new primes are the factors:

    >>> factorint(primorial(4) - 1)
    {11: 1, 19: 1}

    Here, some primes smaller and larger than the primes multiplied together
    are obtained:

    >>> p = list(primerange(10, 20))
    >>> sorted(set(primefactors(Mul(*p) + 1)).difference(set(p)))
    [2, 5, 31, 149]

    See Also
    ========

    primerange : Generate all primes in a given range

    """
    if nth:
        n = as_int(n)
    else:
        n = int(n)
    if n < 1:
        raise ValueError('primorial argument must be >= 1')
    p = 1
    if nth:
        for i in range(1, n + 1):
            p *= prime(i)
    else:
        for i in primerange(2, n + 1):
            p *= i
    return p