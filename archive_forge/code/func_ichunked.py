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
def ichunked(iterable, n):
    """Break *iterable* into sub-iterables with *n* elements each.
    :func:`ichunked` is like :func:`chunked`, but it yields iterables
    instead of lists.

    If the sub-iterables are read in order, the elements of *iterable*
    won't be stored in memory.
    If they are read out of order, :func:`itertools.tee` is used to cache
    elements as necessary.

    >>> from itertools import count
    >>> all_chunks = ichunked(count(), 4)
    >>> c_1, c_2, c_3 = next(all_chunks), next(all_chunks), next(all_chunks)
    >>> list(c_2)  # c_1's elements have been cached; c_3's haven't been
    [4, 5, 6, 7]
    >>> list(c_1)
    [0, 1, 2, 3]
    >>> list(c_3)
    [8, 9, 10, 11]

    """
    source = peekable(iter(iterable))
    ichunk_marker = object()
    while True:
        item = source.peek(ichunk_marker)
        if item is ichunk_marker:
            return
        chunk = _IChunk(source, n)
        yield chunk
        chunk.fill_cache()