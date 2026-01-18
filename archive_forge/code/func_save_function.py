import _collections_abc
import abc
import copyreg
import io
import itertools
import logging
import sys
import struct
import types
import weakref
import typing
from enum import Enum
from collections import ChainMap, OrderedDict
from .compat import pickle, Pickler
from .cloudpickle import (
def save_function(self, obj, name=None):
    """Registered with the dispatch to handle all function types.

            Determines what kind of function obj is (e.g. lambda, defined at
            interactive prompt, etc) and handles the pickling appropriately.
            """
    if _should_pickle_by_reference(obj, name=name):
        return Pickler.save_global(self, obj, name=name)
    elif PYPY and isinstance(obj.__code__, builtin_code_type):
        return self.save_pypy_builtin_func(obj)
    else:
        return self._save_reduce_pickle5(*self._dynamic_function_reduce(obj), obj=obj)