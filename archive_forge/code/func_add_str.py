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
def add_str(self):
    repr = self._cls_dict.get('__repr__')
    if repr is None:
        msg = '__str__ can only be generated if a __repr__ exists.'
        raise ValueError(msg)

    def __str__(self):
        return self.__repr__()
    self._cls_dict['__str__'] = self._add_method_dunders(__str__)
    return self