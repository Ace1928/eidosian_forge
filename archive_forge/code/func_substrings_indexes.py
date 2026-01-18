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
def substrings_indexes(seq, reverse=False):
    """Yield all substrings and their positions in *seq*

    The items yielded will be a tuple of the form ``(substr, i, j)``, where
    ``substr == seq[i:j]``.

    This function only works for iterables that support slicing, such as
    ``str`` objects.

    >>> for item in substrings_indexes('more'):
    ...    print(item)
    ('m', 0, 1)
    ('o', 1, 2)
    ('r', 2, 3)
    ('e', 3, 4)
    ('mo', 0, 2)
    ('or', 1, 3)
    ('re', 2, 4)
    ('mor', 0, 3)
    ('ore', 1, 4)
    ('more', 0, 4)

    Set *reverse* to ``True`` to yield the same items in the opposite order.


    """
    r = range(1, len(seq) + 1)
    if reverse:
        r = reversed(r)
    return ((seq[i:i + L], i, i + L) for L in r for i in range(len(seq) - L + 1))