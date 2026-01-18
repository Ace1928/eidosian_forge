import re
import operator
import heapq
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cached_property
from numba.core import config
import llvmlite.binding as llvm
def make_adjuster(attr):
    time_attr = f'{attr}_time'
    percent_attr = f'{attr}_percent'
    time_getter = operator.attrgetter(time_attr)

    def adjust(d):
        """Compute percent x total_time = adjusted"""
        total = time_getter(total_rec)
        adjusted = total * d[percent_attr] * 0.01
        d[time_attr] = adjusted
        return d
    return adjust