import random
from bisect import bisect
from itertools import count
from array import array as _array
from sympy.core.function import Function
from sympy.core.singleton import S
from .primetest import isprime
from sympy.utilities.misc import as_int
def mobiusrange(self, a, b):
    """Generate all mobius numbers for the range [a, b).

        Parameters
        ==========

        a : integer
            First number in range

        b : integer
            First number outside of range

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.mobiusrange(7, 18)])
        [-1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1]
        """
    a = max(1, _as_int_ceiling(a))
    b = _as_int_ceiling(b)
    n = len(self._mlist)
    if a >= b:
        return
    elif b <= n:
        for i in range(a, b):
            yield self._mlist[i]
    else:
        self._mlist += _azeros(b - n)
        for i in range(1, n):
            mi = self._mlist[i]
            startindex = (n + i - 1) // i * i
            for j in range(startindex, b, i):
                self._mlist[j] -= mi
            if i >= a:
                yield mi
        for i in range(n, b):
            mi = self._mlist[i]
            for j in range(2 * i, b, i):
                self._mlist[j] -= mi
            if i >= a:
                yield mi