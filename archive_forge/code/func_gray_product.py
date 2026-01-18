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
def gray_product(*iterables):
    """Like :func:`itertools.product`, but return tuples in an order such
    that only one element in the generated tuple changes from one iteration
    to the next.

        >>> list(gray_product('AB','CD'))
        [('A', 'C'), ('B', 'C'), ('B', 'D'), ('A', 'D')]

    This function consumes all of the input iterables before producing output.
    If any of the input iterables have fewer than two items, ``ValueError``
    is raised.

    For information on the algorithm, see
    `this section <https://www-cs-faculty.stanford.edu/~knuth/fasc2a.ps.gz>`__
    of Donald Knuth's *The Art of Computer Programming*.
    """
    all_iterables = tuple((tuple(x) for x in iterables))
    iterable_count = len(all_iterables)
    for iterable in all_iterables:
        if len(iterable) < 2:
            raise ValueError('each iterable must have two or more items')
    a = [0] * iterable_count
    f = list(range(iterable_count + 1))
    o = [1] * iterable_count
    while True:
        yield tuple((all_iterables[i][a[i]] for i in range(iterable_count)))
        j = f[0]
        f[0] = 0
        if j == iterable_count:
            break
        a[j] = a[j] + o[j]
        if a[j] == 0 or a[j] == len(all_iterables[j]) - 1:
            o[j] = -o[j]
            f[j] = f[j + 1]
            f[j + 1] = j + 1