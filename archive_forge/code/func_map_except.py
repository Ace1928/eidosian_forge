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
def map_except(function, iterable, *exceptions):
    """Transform each item from *iterable* with *function* and yield the
    result, unless *function* raises one of the specified *exceptions*.

    *function* is called to transform each item in *iterable*.
    It should accept one argument.

    >>> iterable = ['1', '2', 'three', '4', None]
    >>> list(map_except(int, iterable, ValueError, TypeError))
    [1, 2, 4]

    If an exception other than one given by *exceptions* is raised by
    *function*, it is raised like normal.
    """
    for item in iterable:
        try:
            yield function(item)
        except exceptions:
            pass