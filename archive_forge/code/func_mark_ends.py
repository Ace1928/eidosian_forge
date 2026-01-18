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
def mark_ends(iterable):
    """Yield 3-tuples of the form ``(is_first, is_last, item)``.

    >>> list(mark_ends('ABC'))
    [(True, False, 'A'), (False, False, 'B'), (False, True, 'C')]

    Use this when looping over an iterable to take special action on its first
    and/or last items:

    >>> iterable = ['Header', 100, 200, 'Footer']
    >>> total = 0
    >>> for is_first, is_last, item in mark_ends(iterable):
    ...     if is_first:
    ...         continue  # Skip the header
    ...     if is_last:
    ...         continue  # Skip the footer
    ...     total += item
    >>> print(total)
    300
    """
    it = iter(iterable)
    try:
        b = next(it)
    except StopIteration:
        return
    try:
        for i in count():
            a = b
            b = next(it)
            yield (i == 0, False, a)
    except StopIteration:
        yield (i == 0, True, a)