from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def assertRegex(self, *args, **kwargs):
    return getattr(self, _assertRegex)(*args, **kwargs)