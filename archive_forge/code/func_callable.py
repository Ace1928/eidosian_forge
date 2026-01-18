from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def callable(obj):
    return any(('__call__' in klass.__dict__ for klass in type(obj).__mro__))