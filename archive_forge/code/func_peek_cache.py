from __future__ import absolute_import, division, print_function
import logging
from functools import wraps, update_wrapper
import types
from warnings import warn
from passlib.utils.compat import PY3
def peek_cache(self, obj, default=None):
    """
        class-level helper to peek at stored value

        usage: :samp:`value = type(self).{attr}.clear_cache(self)`
        """
    return obj.__dict__.get(self.__name__, default)