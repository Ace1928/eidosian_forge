import random
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from sympy.core.parameters import global_parameters
from sympy.core.basic import Atom
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.sympify import _sympify
from sympy.matrices import zeros
from sympy.polys.polytools import lcm
from sympy.printing.repr import srepr
from sympy.utilities.iterables import (flatten, has_variety, minlex,
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import ifac
from sympy.multipledispatch import dispatch
def inversions(self):
    """
        Computes the number of inversions of a permutation.

        Explanation
        ===========

        An inversion is where i > j but p[i] < p[j].

        For small length of p, it iterates over all i and j
        values and calculates the number of inversions.
        For large length of p, it uses a variation of merge
        sort to calculate the number of inversions.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3, 4, 5])
        >>> p.inversions()
        0
        >>> Permutation([3, 2, 1, 0]).inversions()
        6

        See Also
        ========

        descents, ascents, min, max

        References
        ==========

        .. [1] https://www.cp.eng.chula.ac.th/~prabhas//teaching/algo/algo2008/count-inv.htm

        """
    inversions = 0
    a = self.array_form
    n = len(a)
    if n < 130:
        for i in range(n - 1):
            b = a[i]
            for c in a[i + 1:]:
                if b > c:
                    inversions += 1
    else:
        k = 1
        right = 0
        arr = a[:]
        temp = a[:]
        while k < n:
            i = 0
            while i + k < n:
                right = i + k * 2 - 1
                if right >= n:
                    right = n - 1
                inversions += _merge(arr, temp, i, i + k, right)
                i = i + k * 2
            k = k * 2
    return inversions