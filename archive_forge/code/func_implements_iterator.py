import types
import sys
import numbers
import functools
import copy
import inspect
def implements_iterator(cls):
    """
    From jinja2/_compat.py. License: BSD.

    Use as a decorator like this::

        @implements_iterator
        class UppercasingIterator(object):
            def __init__(self, iterable):
                self._iter = iter(iterable)
            def __iter__(self):
                return self
            def __next__(self):
                return next(self._iter).upper()

    """
    if PY3:
        return cls
    else:
        cls.next = cls.__next__
        del cls.__next__
        return cls