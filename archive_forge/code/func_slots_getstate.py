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
def slots_getstate(self):
    """
            Automatically created by attrs.
            """
    return {name: getattr(self, name) for name in state_attr_names}