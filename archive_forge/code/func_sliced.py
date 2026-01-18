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
def sliced(seq, n, strict=False):
    """Yield slices of length *n* from the sequence *seq*.

    >>> list(sliced((1, 2, 3, 4, 5, 6), 3))
    [(1, 2, 3), (4, 5, 6)]

    By the default, the last yielded slice will have fewer than *n* elements
    if the length of *seq* is not divisible by *n*:

    >>> list(sliced((1, 2, 3, 4, 5, 6, 7, 8), 3))
    [(1, 2, 3), (4, 5, 6), (7, 8)]

    If the length of *seq* is not divisible by *n* and *strict* is
    ``True``, then ``ValueError`` will be raised before the last
    slice is yielded.

    This function will only work for iterables that support slicing.
    For non-sliceable iterables, see :func:`chunked`.

    """
    iterator = takewhile(len, (seq[i:i + n] for i in count(0, n)))
    if strict:

        def ret():
            for _slice in iterator:
                if len(_slice) != n:
                    raise ValueError('seq is not divisible by n.')
                yield _slice
        return iter(ret())
    else:
        return iterator