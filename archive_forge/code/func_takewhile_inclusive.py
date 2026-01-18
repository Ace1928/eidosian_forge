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
def takewhile_inclusive(predicate, iterable):
    """A variant of :func:`takewhile` that yields one additional element.

        >>> list(takewhile_inclusive(lambda x: x < 5, [1, 4, 6, 4, 1]))
        [1, 4, 6]

    :func:`takewhile` would return ``[1, 4]``.
    """
    for x in iterable:
        yield x
        if not predicate(x):
            break