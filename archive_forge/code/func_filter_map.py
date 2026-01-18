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
def filter_map(func, iterable):
    """Apply *func* to every element of *iterable*, yielding only those which
    are not ``None``.

    >>> elems = ['1', 'a', '2', 'b', '3']
    >>> list(filter_map(lambda s: int(s) if s.isnumeric() else None, elems))
    [1, 2, 3]
    """
    for x in iterable:
        y = func(x)
        if y is not None:
            yield y