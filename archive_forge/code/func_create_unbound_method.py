from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def create_unbound_method(func, cls):
    return types.MethodType(func, None, cls)