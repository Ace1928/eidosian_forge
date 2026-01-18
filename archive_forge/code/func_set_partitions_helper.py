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
def set_partitions_helper(L, k):
    n = len(L)
    if k == 1:
        yield [L]
    elif n == k:
        yield [[s] for s in L]
    else:
        e, *M = L
        for p in set_partitions_helper(M, k - 1):
            yield [[e], *p]
        for p in set_partitions_helper(M, k):
            for i in range(len(p)):
                yield (p[:i] + [[e] + p[i]] + p[i + 1:])