import random
from bisect import bisect
from itertools import count
from array import array as _array
from sympy.core.function import Function
from sympy.core.singleton import S
from .primetest import isprime
from sympy.utilities.misc import as_int
def extend_to_no(self, i):
    """Extend to include the ith prime number.

        Parameters
        ==========

        i : integer

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend_to_no(9)
        >>> sieve._list
        array('l', [2, 3, 5, 7, 11, 13, 17, 19, 23])

        Notes
        =====

        The list is extended by 50% if it is too short, so it is
        likely that it will be longer than requested.
        """
    i = as_int(i)
    while len(self._list) < i:
        self.extend(int(self._list[-1] * 1.5))