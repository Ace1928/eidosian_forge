import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import functools
import builtins
from keyword import iskeyword
from operator import attrgetter
from collections import namedtuple, OrderedDict
def isawaitable(object):
    """Return true if object can be passed to an ``await`` expression."""
    return isinstance(object, types.CoroutineType) or (isinstance(object, types.GeneratorType) and bool(object.gi_code.co_flags & CO_ITERABLE_COROUTINE)) or isinstance(object, collections.abc.Awaitable)