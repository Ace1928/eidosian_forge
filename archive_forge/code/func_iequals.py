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
def iequals(*iterables):
    """Return ``True`` if all given *iterables* are equal to each other,
    which means that they contain the same elements in the same order.

    The function is useful for comparing iterables of different data types
    or iterables that do not support equality checks.

    >>> iequals("abc", ['a', 'b', 'c'], ('a', 'b', 'c'), iter("abc"))
    True

    >>> iequals("abc", "acb")
    False

    Not to be confused with :func:`all_equals`, which checks whether all
    elements of iterable are equal to each other.

    """
    return all(map(all_equal, zip_longest(*iterables, fillvalue=object())))