import warnings
from collections import Counter, defaultdict, deque, abc
from collections.abc import Sequence
from functools import cached_property, partial, reduce, wraps
from heapq import heapify, heapreplace, heappop
from itertools import (
from math import exp, factorial, floor, log, perm, comb
from queue import Empty, Queue
from random import random, randrange, uniform
from operator import itemgetter, mul, sub, gt, lt, ge, le
from sys import hexversion, maxsize
from time import monotonic
from .recipes import (
def combination_with_replacement_index(element, iterable):
    """Equivalent to
    ``list(combinations_with_replacement(iterable, r)).index(element)``

    The subsequences with repetition of *iterable* that are of length *r* can
    be ordered lexicographically. :func:`combination_with_replacement_index`
    computes the index of the first *element*, without computing the previous
    combinations with replacement.

        >>> combination_with_replacement_index('adf', 'abcdefg')
        20

    ``ValueError`` will be raised if the given *element* isn't one of the
    combinations with replacement of *iterable*.
    """
    element = tuple(element)
    l = len(element)
    element = enumerate(element)
    k, y = next(element, (None, None))
    if k is None:
        return 0
    indexes = []
    pool = tuple(iterable)
    for n, x in enumerate(pool):
        while x == y:
            indexes.append(n)
            tmp, y = next(element, (None, None))
            if tmp is None:
                break
            else:
                k = tmp
        if y is None:
            break
    else:
        raise ValueError('element is not a combination with replacement of iterable')
    n = len(pool)
    occupations = [0] * n
    for p in indexes:
        occupations[p] += 1
    index = 0
    cumulative_sum = 0
    for k in range(1, n):
        cumulative_sum += occupations[k - 1]
        j = l + n - 1 - k - cumulative_sum
        i = n - k
        if i <= j:
            index += comb(j, i)
    return index