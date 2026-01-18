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
def add_match_args(self):
    self._cls_dict['__match_args__'] = tuple((field.name for field in self._attrs if field.init and (not field.kw_only)))