import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def intersect_counts(counts1, counts2):
    d = {}
    for k, v1 in counts1.items():
        if k in counts2:
            d[k] = min(v1, counts2[k])
    return d