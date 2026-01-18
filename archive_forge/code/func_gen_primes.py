import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def gen_primes():
    d = defaultdict(list)
    q = 2
    while 1:
        if q not in d:
            yield q
            d[q * q].append(q)
        else:
            for p in d[q]:
                d[p + q].append(p)
            del d[q]
        q += 1