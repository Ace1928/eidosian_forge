from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
def cumsums(self, *items):
    total = None
    sums = []
    for item in items:
        if total is None:
            total = item
        else:
            total += item
        sums.append(total)
    return sums