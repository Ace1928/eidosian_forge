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
def apply_defaults(self):
    """Set default values for missing arguments.

        For variable-positional arguments (*args) the default is an
        empty tuple.

        For variable-keyword arguments (**kwargs) the default is an
        empty dict.
        """
    arguments = self.arguments
    new_arguments = []
    for name, param in self._signature.parameters.items():
        try:
            new_arguments.append((name, arguments[name]))
        except KeyError:
            if param.default is not _empty:
                val = param.default
            elif param.kind is _VAR_POSITIONAL:
                val = ()
            elif param.kind is _VAR_KEYWORD:
                val = {}
            else:
                continue
            new_arguments.append((name, val))
    self.arguments = dict(new_arguments)