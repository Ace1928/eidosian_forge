from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def raise_from(value, from_value):
    raise value