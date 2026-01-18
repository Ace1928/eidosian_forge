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
def distinct_permutations(iterable, r=None):
    """Yield successive distinct permutations of the elements in *iterable*.

        >>> sorted(distinct_permutations([1, 0, 1]))
        [(0, 1, 1), (1, 0, 1), (1, 1, 0)]

    Equivalent to ``set(permutations(iterable))``, except duplicates are not
    generated and thrown away. For larger input sequences this is much more
    efficient.

    Duplicate permutations arise when there are duplicated elements in the
    input iterable. The number of items returned is
    `n! / (x_1! * x_2! * ... * x_n!)`, where `n` is the total number of
    items input, and each `x_i` is the count of a distinct item in the input
    sequence.

    If *r* is given, only the *r*-length permutations are yielded.

        >>> sorted(distinct_permutations([1, 0, 1], r=2))
        [(0, 1), (1, 0), (1, 1)]
        >>> sorted(distinct_permutations(range(3), r=2))
        [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

    """

    def _full(A):
        while True:
            yield tuple(A)
            for i in range(size - 2, -1, -1):
                if A[i] < A[i + 1]:
                    break
            else:
                return
            for j in range(size - 1, i, -1):
                if A[i] < A[j]:
                    break
            A[i], A[j] = (A[j], A[i])
            A[i + 1:] = A[:i - size:-1]

    def _partial(A, r):
        head, tail = (A[:r], A[r:])
        right_head_indexes = range(r - 1, -1, -1)
        left_tail_indexes = range(len(tail))
        while True:
            yield tuple(head)
            pivot = tail[-1]
            for i in right_head_indexes:
                if head[i] < pivot:
                    break
                pivot = head[i]
            else:
                return
            for j in left_tail_indexes:
                if tail[j] > head[i]:
                    head[i], tail[j] = (tail[j], head[i])
                    break
            else:
                for j in right_head_indexes:
                    if head[j] > head[i]:
                        head[i], head[j] = (head[j], head[i])
                        break
            tail += head[:i - r:-1]
            i += 1
            head[i:], tail[:] = (tail[:r - i], tail[r - i:])
    items = sorted(iterable)
    size = len(items)
    if r is None:
        r = size
    if 0 < r <= size:
        return _full(items) if r == size else _partial(items, r)
    return iter(() if r else ((),))