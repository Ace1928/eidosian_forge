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
def adjacent(predicate, iterable, distance=1):
    """Return an iterable over `(bool, item)` tuples where the `item` is
    drawn from *iterable* and the `bool` indicates whether
    that item satisfies the *predicate* or is adjacent to an item that does.

    For example, to find whether items are adjacent to a ``3``::

        >>> list(adjacent(lambda x: x == 3, range(6)))
        [(False, 0), (False, 1), (True, 2), (True, 3), (True, 4), (False, 5)]

    Set *distance* to change what counts as adjacent. For example, to find
    whether items are two places away from a ``3``:

        >>> list(adjacent(lambda x: x == 3, range(6), distance=2))
        [(False, 0), (True, 1), (True, 2), (True, 3), (True, 4), (True, 5)]

    This is useful for contextualizing the results of a search function.
    For example, a code comparison tool might want to identify lines that
    have changed, but also surrounding lines to give the viewer of the diff
    context.

    The predicate function will only be called once for each item in the
    iterable.

    See also :func:`groupby_transform`, which can be used with this function
    to group ranges of items with the same `bool` value.

    """
    if distance < 0:
        raise ValueError('distance must be at least 0')
    i1, i2 = tee(iterable)
    padding = [False] * distance
    selected = chain(padding, map(predicate, i1), padding)
    adjacent_to_selected = map(any, windowed(selected, 2 * distance + 1))
    return zip(adjacent_to_selected, i2)