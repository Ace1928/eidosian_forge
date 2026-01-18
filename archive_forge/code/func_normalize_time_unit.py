from __future__ import print_function, division, absolute_import
import ctypes
import operator
from collections import OrderedDict
from math import ceil
from datashader import datashape
import numpy as np
from .internal_utils import IndexCallable, isidentifier
def normalize_time_unit(s):
    """ Normalize time input to one of 'year', 'second', 'millisecond', etc..
    Example
    -------
    >>> normalize_time_unit('milliseconds')
    'ms'
    >>> normalize_time_unit('ms')
    'ms'
    >>> normalize_time_unit('nanoseconds')
    'ns'
    >>> normalize_time_unit('nanosecond')
    'ns'
    """
    s = s.strip()
    if s in _units:
        return s
    if s in _unit_aliases:
        return _unit_aliases[s]
    if s[-1] == 's' and len(s) > 2:
        return normalize_time_unit(s.rstrip('s'))
    raise ValueError('Do not understand time unit %s' % s)