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
def random_integer_partition(n, seed=None):
    """
    Generates a random integer partition summing to ``n`` as a list
    of reverse-sorted integers.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import random_integer_partition

    For the following, a seed is given so a known value can be shown; in
    practice, the seed would not be given.

    >>> random_integer_partition(100, seed=[1, 1, 12, 1, 2, 1, 85, 1])
    [85, 12, 2, 1]
    >>> random_integer_partition(10, seed=[1, 2, 3, 1, 5, 1])
    [5, 3, 1, 1]
    >>> random_integer_partition(1)
    [1]
    """
    from sympy.core.random import _randint
    n = as_int(n)
    if n < 1:
        raise ValueError('n must be a positive integer')
    randint = _randint(seed)
    partition = []
    while n > 0:
        k = randint(1, n)
        mult = randint(1, n // k)
        partition.append((k, mult))
        n -= k * mult
    partition.sort(reverse=True)
    partition = flatten([[k] * m for k, m in partition])
    return partition