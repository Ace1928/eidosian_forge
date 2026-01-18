import warnings
from collections import Counter, defaultdict, deque, abc
from collections.abc import Sequence
from functools import partial, reduce, wraps
from heapq import heapify, heapreplace, heappop
from itertools import (
from math import exp, factorial, floor, log
from queue import Empty, Queue
from random import random, randrange, uniform
from operator import itemgetter, mul, sub, gt, lt, ge, le
from sys import hexversion, maxsize
from time import monotonic
from .recipes import (
def permutation_index(element, iterable):
    """Equivalent to ``list(permutations(iterable, r)).index(element)```

    The subsequences of *iterable* that are of length *r* where order is
    important can be ordered lexicographically. :func:`permutation_index`
    computes the index of the first *element* directly, without computing
    the previous permutations.

        >>> permutation_index([1, 3, 2], range(5))
        19

    ``ValueError`` will be raised if the given *element* isn't one of the
    permutations of *iterable*.
    """
    index = 0
    pool = list(iterable)
    for i, x in zip(range(len(pool), -1, -1), element):
        r = pool.index(x)
        index = index * i + r
        del pool[r]
    return index