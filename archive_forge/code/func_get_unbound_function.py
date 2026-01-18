from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def get_unbound_function(unbound):
    return unbound.im_func