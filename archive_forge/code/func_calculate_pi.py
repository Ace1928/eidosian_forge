from collections import deque
from functools import reduce
from math import ceil, floor
import operator
import re
from itertools import chain
import six
from genshi.compat import IS_PYTHON2
from genshi.core import Stream, Attrs, Namespace, QName
from genshi.core import START, END, TEXT, START_NS, END_NS, COMMENT, PI, \
def calculate_pi(f):
    """KMP prefix calculation for table"""
    if len(f) == 0:
        return []
    pi = [0]
    s = 0
    for i in range(1, len(f)):
        while s > 0 and (not nodes_equal(f[s], f[i])):
            s = pi[s - 1]
        if nodes_equal(f[s], f[i]):
            s += 1
        pi.append(s)
    return pi