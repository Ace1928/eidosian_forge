import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def decide_callable_or_boolean(value):
    """
        Decide whether a key function is used.
        """
    if callable(value):
        value, key = (True, value)
    else:
        key = None
    return (value, key)