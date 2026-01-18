from sympy.core import Basic, Dict, sympify, Tuple
from sympy.core.numbers import Integer
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.numbers import bell
from sympy.matrices import zeros
from sympy.sets.sets import FiniteSet, Union
from sympy.utilities.iterables import flatten, group
from sympy.utilities.misc import as_int
from collections import defaultdict
def next_lex(self):
    """Return the next partition of the integer, n, in lexical order,
        wrapping around to [n] if the partition is [1, ..., 1].

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> p = IntegerPartition([3, 1])
        >>> print(p.next_lex())
        [4]
        >>> p.partition < p.next_lex().partition
        True
        """
    d = defaultdict(int)
    d.update(self.as_dict())
    key = self._keys
    a = key[-1]
    if a == self.integer:
        d.clear()
        d[1] = self.integer
    elif a == 1:
        if d[a] > 1:
            d[a + 1] += 1
            d[a] -= 2
        else:
            b = key[-2]
            d[b + 1] += 1
            d[1] = (d[b] - 1) * b
            d[b] = 0
    elif d[a] > 1:
        if len(key) == 1:
            d.clear()
            d[a + 1] = 1
            d[1] = self.integer - a - 1
        else:
            a1 = a + 1
            d[a1] += 1
            d[1] = d[a] * a - a1
            d[a] = 0
    else:
        b = key[-2]
        b1 = b + 1
        d[b1] += 1
        need = d[b] * b + d[a] * a - b1
        d[a] = d[b] = 0
        d[1] = need
    return IntegerPartition(self.integer, d)