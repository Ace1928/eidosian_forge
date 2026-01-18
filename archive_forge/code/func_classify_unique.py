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
def classify_unique(iterable, key=None):
    """Classify each element in terms of its uniqueness.

    For each element in the input iterable, return a 3-tuple consisting of:

    1. The element itself
    2. ``False`` if the element is equal to the one preceding it in the input,
       ``True`` otherwise (i.e. the equivalent of :func:`unique_justseen`)
    3. ``False`` if this element has been seen anywhere in the input before,
       ``True`` otherwise (i.e. the equivalent of :func:`unique_everseen`)

    >>> list(classify_unique('otto'))    # doctest: +NORMALIZE_WHITESPACE
    [('o', True,  True),
     ('t', True,  True),
     ('t', False, False),
     ('o', True,  False)]

    This function is analogous to :func:`unique_everseen` and is subject to
    the same performance considerations.

    """
    seen_set = set()
    seen_list = []
    use_key = key is not None
    previous = None
    for i, element in enumerate(iterable):
        k = key(element) if use_key else element
        is_unique_justseen = not i or previous != k
        previous = k
        is_unique_everseen = False
        try:
            if k not in seen_set:
                seen_set.add(k)
                is_unique_everseen = True
        except TypeError:
            if k not in seen_list:
                seen_list.append(k)
                is_unique_everseen = True
        yield (element, is_unique_justseen, is_unique_everseen)