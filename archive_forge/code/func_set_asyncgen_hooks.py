import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
def set_asyncgen_hooks(firstiter=UNSPECIFIED, finalizer=UNSPECIFIED):
    if firstiter is not UNSPECIFIED:
        if firstiter is None or callable(firstiter):
            _hooks.firstiter = firstiter
        else:
            raise TypeError('callable firstiter expected, got {}'.format(type(firstiter).__name__))
    if finalizer is not UNSPECIFIED:
        if finalizer is None or callable(finalizer):
            _hooks.finalizer = finalizer
        else:
            raise TypeError('callable finalizer expected, got {}'.format(type(finalizer).__name__))