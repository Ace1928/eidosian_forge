import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def parse_select(s):
    ranges = []
    start = 0
    while 1:
        m = range_pat.match(s, start)
        if m is not None:
            left = int(m.group(1))
            right = m.group(2)
            if not right:
                ranges.append(starting_from(left - 1))
            else:
                ranges.append(range(left - 1, int(right)))
        else:
            m = value_pat.match(s, start)
            if m is not None:
                val = int(m.group(1))
                ranges.append(range(val - 1, val))
            else:
                raise argparse.ArgumentTypeError('Unknown character at position %d of %r' % (start + 1, s))
        start = m.end()
        t = s[start:start + 1]
        if not t:
            break
        if t == ',':
            start += 1
            continue
        raise argparse.ArgumentTypeError('Unknown character at position %d of %r' % (start + 1, s))
    return ranges