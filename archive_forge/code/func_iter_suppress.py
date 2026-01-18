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
def iter_suppress(iterable, *exceptions):
    """Yield each of the items from *iterable*. If the iteration raises one of
    the specified *exceptions*, that exception will be suppressed and iteration
    will stop.

    >>> from itertools import chain
    >>> def breaks_at_five(x):
    ...     while True:
    ...         if x >= 5:
    ...             raise RuntimeError
    ...         yield x
    ...         x += 1
    >>> it_1 = iter_suppress(breaks_at_five(1), RuntimeError)
    >>> it_2 = iter_suppress(breaks_at_five(2), RuntimeError)
    >>> list(chain(it_1, it_2))
    [1, 2, 3, 4, 2, 3, 4]
    """
    try:
        yield from iterable
    except exceptions:
        return