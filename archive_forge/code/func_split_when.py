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
def split_when(iterable, pred, maxsplit=-1):
    """Split *iterable* into pieces based on the output of *pred*.
    *pred* should be a function that takes successive pairs of items and
    returns ``True`` if the iterable should be split in between them.

    For example, to find runs of increasing numbers, split the iterable when
    element ``i`` is larger than element ``i + 1``:

        >>> list(split_when([1, 2, 3, 3, 2, 5, 2, 4, 2], lambda x, y: x > y))
        [[1, 2, 3, 3], [2, 5], [2, 4], [2]]

    At most *maxsplit* splits are done. If *maxsplit* is not specified or -1,
    then there is no limit on the number of splits:

        >>> list(split_when([1, 2, 3, 3, 2, 5, 2, 4, 2],
        ...                 lambda x, y: x > y, maxsplit=2))
        [[1, 2, 3, 3], [2, 5], [2, 4, 2]]

    """
    if maxsplit == 0:
        yield list(iterable)
        return
    it = iter(iterable)
    try:
        cur_item = next(it)
    except StopIteration:
        return
    buf = [cur_item]
    for next_item in it:
        if pred(cur_item, next_item):
            yield buf
            if maxsplit == 1:
                yield ([next_item] + list(it))
                return
            buf = []
            maxsplit -= 1
        buf.append(next_item)
        cur_item = next_item
    yield buf