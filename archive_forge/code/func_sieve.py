import math
import operator
import warnings
from collections import deque
from collections.abc import Sized
from functools import reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
def sieve(n):
    """Yield the primes less than n.

    >>> list(sieve(30))
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    """
    isqrt = getattr(math, 'isqrt', lambda x: int(math.sqrt(x)))
    data = bytearray((0, 1)) * (n // 2)
    data[:3] = (0, 0, 0)
    limit = isqrt(n) + 1
    for p in compress(range(limit), data):
        data[p * p:n:p + p] = bytes(len(range(p * p, n, p + p)))
    data[2] = 1
    return iter_index(data, 1) if n > 2 else iter([])